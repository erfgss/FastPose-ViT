> Current extension (2026-11-07): optional geometry dual-token decoder, frozen pretraining and joint training passed on NPU. See [Geometry_dual_token.md](Geometry_dual_token.md) for exact changes, evidence, commands and limitations. Baseline retained; no demonstrated accuracy improvement.

> Update 2026-11-07: real SPEED validation is now complete; see [SPEED_validation.md](SPEED_validation.md). Any missing-SPEED or synthetic-only status below describes the earlier migration stage. SPEED+ remains unavailable.

# Ascend NPU usage

Validated on 2026-11-06: Ascend 910B2C (64GB), driver 26.0.rc1, CANN 9.0.0, Python 3.11.15, torch 2.10.0+cpu, torch_npu 2.10.0, torchvision 0.25.0+cpu. Environment gate: AUTO. Isolated .venv-npu; system framework preserved.

## Activate and verify
```bash
cd /data/FastPose-ViT
source .venv-npu/bin/activate
export FASTPOSE_DEVICE=npu:0
PYTHONPATH=. python scripts/smoke_npu.py
PYTHONPATH=. python scripts/check_npu_attention.py
```
`FASTPOSE_DEVICE=auto` (default) selects NPU, then CUDA, then CPU. Set `cpu`, `cuda:0`, or `npu:0` explicitly to override.

## Recreate the Python environment
Use the existing CANN-enabled Python with matching torch and torch_npu already installed:
```bash
python -m venv --system-site-packages .venv-npu
source .venv-npu/bin/activate
pip install --no-deps torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-npu.txt
pip install -e . --no-deps
```
Do not use requirements-pytorch.txt or the CUDA requirements.txt in this environment. Do not compile the CUDA MSDA extension. NPU uses differentiable PyTorch grid_sample attention. torch.compile CUDA/Inductor modes are skipped on NPU; no global CUDA API monkeypatches are used.

## Data and real evaluation
SPEED/SPEED+ is absent on this server. Download from the official links in the root README (SPEED: https://zenodo.org/records/6327547 ; SPEED+: https://zenodo.org/records/5588480), prepare splits and bounding-box JSON as documented there, then use the existing arguments with these NPU entrypoints:
```bash
python -m src.train --help
python -m src.evaluate --help
python -m src.predict --help
python -m object_detector.train --help
python -m object_detector.evaluate --help
python -m optimization.evaluate_torch_latency --help
```
Example after the SPEED_FIXED data is prepared under /data:
```bash
python -m src.evaluate --dataset SPEED --dataset_root_dir /data --vit_model vit_b_16_no_weights --model_weights weights/vit_b_16.pth --rotation_format matrix --merge_outputs --no_mlp
```
Use the no_weights variants when loading a full pose checkpoint to avoid unnecessary ImageNet downloads. Provided pose checkpoints use a merged 9-value head with no MLP (translation 3, rotation 6). NCHW float32 inputs: Bx3x224x224 or Bx3x384x384. Detector smoke uses Bx3x640x640; output boxes Bx1x4 and logits Bx1x1.

## Baselines and limits
The smoke uses seeded synthetic inputs and supplied trained checkpoints, with CPU FP32 as a numerical reference. GPU performance baseline is unavailable (N/A); run the same benchmark on a CUDA host using FASTPOSE_DEVICE=cuda:0 to collect one. No business accuracy or speedup claim is made. Full SPEED pose metrics, real training convergence, target batches, multi-card HCCL and detector AMP need separate validation. NVIDIA TensorRT/PyCUDA scripts remain NVIDIA-specific; an Ascend OM/ACL export is not implemented. Style augmentation checkpoint files are also absent.
Training short-test, once data is present: confirm finite losses and gradients, stop after stable 30-50% loss decrease; do not treat synthetic squared-output loss as convergence evidence.
