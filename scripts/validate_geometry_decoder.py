"""Functional NPU validation, NOT a trained-model accuracy benchmark.

Run from repository root: PYTHONPATH=. FASTPOSE_DEVICE=npu:0
  .venv-npu/bin/python scripts/validate_geometry_decoder.py
"""
import argparse
import gc
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets import SPEEDDataset
from src.device import get_device, make_grad_scaler, synchronize
from src.geometry_decoder import encode_bbox_geometry, forward_pose, GeometryDualTokenDecoder
from src.losses import RelativeTranslationLoss, RelativeRotation6DFrobeniusLoss
from src.models import FastPoseViT
from src.utils import rotation_matrix_to_6d, get_rotation_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/data/datasets/SPEED_FIXED")
    parser.add_argument("--output_dir", default="mig_docs/working/dual_token")
    parser.add_argument("--pretrain_steps", type=int, default=40)
    parser.add_argument("--joint_steps", type=int, default=8)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = get_device()
    if device.type != "npu":
        raise RuntimeError("This validation must run on an Ascend NPU")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"device": str(device), "purpose": "functional validation and tiny-batch overfit"}
    opts = SimpleNamespace(no_pixel_augmentation=True, no_spatial_augmentation=True,
                           domain_gap_pixel_augmentation=False, no_rotation_compensation=False)
    trainset = SPEEDDataset(args.dataset_root, split="train", rotation_format="matrix", args=opts,
                            bbox_json_path=str(Path(args.dataset_root)/"speed_bbox_annotations.json"))
    x, t, r, boxes = next(iter(DataLoader(Subset(trainset, range(4)), batch_size=4)))
    geometry = encode_bbox_geometry(boxes, trainset.camera)
    # Explicit formula check and invalid geometry tests.
    first = boxes[0]
    camera = trainset.camera
    expected = torch.tensor([((first[0]+first[2])/2-camera.cx)/camera.fx,
                             ((first[1]+first[3])/2-camera.cy)/camera.fy,
                             (first[2]-first[0])/camera.fx, (first[3]-first[1])/camera.fy])
    torch.testing.assert_close(geometry[0], expected)
    for invalid in [torch.zeros_like(boxes), boxes * float("nan")]:
        try:
            encode_bbox_geometry(invalid, camera)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid bbox was not rejected")

    # Exact baseline regression against outputs captured before changing source.
    before = torch.load(output_dir/"baseline_before.pt", map_location="cpu", weights_only=True)
    baseline = FastPoseViT("vit_b_16_no_weights", 1, 768, 3, 6, merge_outputs=True, no_mlp=True)
    baseline.load_from_pretrained("weights/vit_b_16.pth")
    result["baseline_parameters"] = sum(p.numel() for p in baseline.parameters())
    baseline.to(device).eval()
    with torch.no_grad():
        after = baseline(before["images"].to(device)).cpu()
    torch.testing.assert_close(after, before["outputs"], atol=0, rtol=0)
    result["baseline_max_abs_change"] = (after-before["outputs"]).abs().max().item()
    del baseline, before, after
    gc.collect()
    torch.npu.empty_cache()

    model = FastPoseViT("vit_b_16_no_weights", 1, 768, 3, 6, merge_outputs=True,
                        no_mlp=True, pose_decoder="geometry_dual_token")
    model.load_from_pretrained("weights/vit_b_16.pth", backbone_only=True)
    result["dual_parameters"] = sum(p.numel() for p in model.parameters())
    result["decoder_parameters"] = sum(p.numel() for p in model.decoder.parameters())
    model.eval()
    with torch.no_grad():
        golden = model(x, geometry)
    model.to(device)
    x, t, r = x.to(device), t.to(device), r.to(device)
    geometry = geometry.to(device)
    with torch.no_grad():
        fp32 = model(x, geometry)
        with torch.autocast("npu", dtype=torch.float16):
            fp16 = model(x, geometry)
        changed_geometry = model(x, geometry + 0.1)
        changed_images = model(x.flip(0), geometry)
        separate = model.decoder(model.backbone.forward_patch_features(x), geometry)
    torch.testing.assert_close(fp32.cpu(), golden, atol=0.005, rtol=0.005)
    torch.testing.assert_close(fp16.float().cpu(), golden, atol=0.01, rtol=0.01)
    torch.testing.assert_close(torch.cat(separate, -1), fp32)
    assert (changed_geometry-fp32).abs().max() > 1e-6
    assert (changed_images-fp32).abs().max() > 1e-6
    result.update(fp32_cpu_max_abs=(fp32.cpu()-golden).abs().max().item(),
                  fp16_cpu_max_abs=(fp16.float().cpu()-golden).abs().max().item(),
                  geometry_response=(changed_geometry-fp32).abs().max().item(),
                  image_response=(changed_images-fp32).abs().max().item())
    del fp32, fp16, changed_geometry, changed_images, separate
    print("Inference, geometry dependence and baseline checks passed", flush=True)

    translation_loss = RelativeTranslationLoss()
    rotation_loss = RelativeRotation6DFrobeniusLoss()
    target6d = rotation_matrix_to_6d(r)

    def loss_fn(pred):
        return translation_loss(pred[:, :3].float(), t) + rotation_loss(pred[:, 3:].float(), target6d)

    def measure_loss():
        model.eval()
        with torch.no_grad():
            return loss_fn(model(x, geometry)).item()

    initial_loss = measure_loss()
    scaler = make_grad_scaler("npu", init_scale=256.)
    phase_results = {}
    for phase, steps, frozen in [("decoder_pretrain", args.pretrain_steps, True),
                                  ("joint_finetune", args.joint_steps, False)]:
        model.set_backbone_trainable(not frozen)
        model.train()
        assert model.backbone.training == (not frozen)
        probe = model.backbone.conv_proj.weight.detach().clone()
        query_probe = model.decoder.task_tokens.detach().clone()
        optimizer = torch.optim.AdamW([
            {"params": model.decoder.parameters(), "lr": 3e-4 if frozen else 5e-5},
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5},
        ])
        losses = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("npu", dtype=torch.float16):
                pred = forward_pose(model, x, boxes, camera)
            loss = loss_fn(pred)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Missing gradient: {name}"
                    assert torch.isfinite(param.grad).all(), f"Nonfinite gradient: {name}"
            if step == 0:
                for name in ["geometry_encoder.0.weight", "task_tokens",
                             "blocks.0.attentions.0.query.weight", "blocks.0.attentions.1.query.weight",
                             "blocks.0.gates.0.weight", "blocks.0.gates.1.weight"]:
                    assert dict(model.decoder.named_parameters())[name].grad.abs().sum() > 0, name
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if step % 10 == 0 or step == steps-1:
                print(f"{phase} step {step+1}/{steps}: loss={loss.item():.6f}", flush=True)
        backbone_delta = (model.backbone.conv_proj.weight-probe).abs().max().item()
        query_delta = (model.decoder.task_tokens-query_probe).abs().max().item()
        assert (backbone_delta == 0) if frozen else (backbone_delta > 0)
        assert query_delta > 0
        phase_results[phase] = {"steps": steps, "losses": losses,
                                "backbone_max_change": backbone_delta, "query_max_change": query_delta,
                                "eval_loss": measure_loss()}
        del optimizer, probe, query_probe
    final_loss = measure_loss()
    result.update(initial_loss=initial_loss, final_loss=final_loss,
                  loss_reduction=1-final_loss/initial_loss, phases=phase_results)
    assert final_loss < initial_loss * 0.7, "Tiny-batch optimization failed to reduce loss by 30%"

    checkpoint_path = output_dir/"geometry_dual_token_smoke.pth"
    model.save(str(checkpoint_path))
    with torch.no_grad():
        expected = model(x, geometry).cpu()
    model.cpu()
    del model
    gc.collect()
    torch.npu.empty_cache()
    restored = FastPoseViT.from_checkpoint(str(checkpoint_path)).to(device).eval()
    with torch.no_grad():
        actual = restored(x, geometry)
        rotations = get_rotation_matrix(actual[:, 3:].float())
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close((rotations.transpose(1, 2) @ rotations).cpu(),
                               torch.eye(3).expand(4, 3, 3), atol=1e-4, rtol=1e-4)
    result["checkpoint_roundtrip_max_abs"] = (actual.cpu()-expected).abs().max().item()
    del restored
    gc.collect()
    torch.npu.empty_cache()

    # Validate the second supported input resolution independently.
    large = FastPoseViT("vit_b_16_384_no_weights", 1, 768, 3, 6, merge_outputs=True,
                        no_mlp=True, pose_decoder="geometry_dual_token")
    large.load_from_pretrained("weights/vit_b_16_384.pth", backbone_only=True)
    large.to(device).eval()
    x384 = torch.nn.functional.interpolate(x[:1], size=(384, 384), mode="bilinear", align_corners=False)
    with torch.no_grad(), torch.autocast("npu", dtype=torch.float16):
        y384 = large(x384, geometry[:1])
    assert y384.shape == (1, 9) and torch.isfinite(y384).all()
    result["resolution_384_fp16"] = "passed (resized real image, inference only)"

    # No-gate ablation is functional and has no orphaned gate parameters.
    ablation = GeometryDualTokenDecoder(32, dim=16, heads=4, layers=1, gated_interaction=False)
    out = ablation(torch.randn(2, 5, 32), torch.randn(2, 4))
    sum(z.square().mean() for z in out).backward()
    assert all(p.grad is not None for p in ablation.parameters())
    result["no_interaction_cpu_backward"] = "passed"
    synchronize(device)
    (output_dir/"results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "phases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
