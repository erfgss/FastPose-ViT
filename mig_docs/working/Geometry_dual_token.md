> Cleanup 20261107_034803: completed smoke weights, CLI prediction outputs and temporary logs were removed with user confirmation. Summary metrics and baseline_before.pt regression reference remain. Current full-training logs and best weights are retained. Historical paths to removed smoke artifacts are no longer live.

# Geometry-conditioned dual-token pose decoder

Implemented and validated on 2026-11-07 in `/data/FastPose-ViT`.
This is an experimental architecture, not a claim of novelty or improved benchmark accuracy.

## Architecture and compatibility

The default `--pose_decoder baseline` retains the original class-token regressor,
its state-dict keys, legacy `vit.*` checkpoint loading, and output format.
Baseline NPU FP32 outputs on four saved real SPEED images matched pre-edit outputs
exactly (maximum absolute difference 0).

`--pose_decoder geometry_dual_token` uses the existing ViT encoder's spatial tokens
(CLS tokens excluded from decoder memory), a 768-to-256 projection, two learned task
queries, and two decoder blocks. Each task has independent multi-head cross-attention
and FFN weights. Each block exchanges information through two sigmoid-gated messages.
`--no_decoder_interaction` disables these messages for an ablation. Default decoder:
256 channels, four attention heads, two blocks. Attention scores/softmax use FP32;
projections and attention values support NPU FP16 autocast. No additional dependencies
or NVIDIA code are introduced.

The geometry encoder maps four dimensionless features into two task-specific query
conditions: `[(u_box-cx)/fx, (v_box-cy)/fy, width_box/fx, height_box/fy]`.
Only bounding boxes and camera intrinsics are used, never pose labels. The geometry
is computed from the final augmented bbox, before crop/resize, in original-image
pixel coordinates. Missing, zero-area and non-finite boxes fail explicitly. Inference
uses the same representation and requires bounding boxes (annotations or detector
predictions). The decoder outputs three relative translation values and a 6D rotation;
existing coordinate restoration and rotation compensation are retained.

The baseline has 85,805,577 parameters; the new model has 90,084,361. The decoder has
4,285,705 parameters, replacing the baseline head (about 5% net parameter increase).
The new decoder supports ViT backbones and 3+6 pose outputs; ConvNeXt/quaternion-head
configurations are rejected. Original baseline options remain available.

## Pretraining and fine-tuning

The supported and tested pretraining path initializes only the ViT backbone from
`weights/vit_b_16.pth`; the new decoder starts with fresh weights. It does not import
the baseline pose head. `--init_backbone_only` makes this behavior explicit.

`--freeze_backbone` freezes backbone gradients AND keeps it in eval mode during
training. Only decoder parameters enter the optimizer. Omitting this option enables
joint fine-tuning. New checkpoints include model weights and architecture metadata;
they do not contain optimizer/scaler state, so loading one starts a new optimization
stage rather than resuming the exact optimizer trajectory. Baseline save format is
unchanged. `FastPoseViT.from_checkpoint(path)` reconstructs new checkpoints; legacy
baseline checkpoints require explicit constructor options.

Two-stage recipe (not launched for full training during this task):

```bash
cd /data/FastPose-ViT
bash scripts/pretrain_geometry_decoder.sh
```

The recipe defaults to 10 frozen-backbone epochs and 30 joint epochs on SPEED_FIXED,
batch size 4, AdamW, FP16 AMP, relative translation and relative 6D rotation losses.
These are starting settings, not tuned hyperparameters. Outputs go to
`experiments/geometry_dual_token/`, preserving existing `weights/` files. Override
`DATASET_ROOT`, `RUN_DIR`, `PRETRAIN_EPOCHS`, and `FINETUNE_EPOCHS` as needed.

Example inference/evaluation using the CLI smoke checkpoint:

```bash
export FASTPOSE_DEVICE=npu:0 OMP_NUM_THREADS=4 PYTHONPATH=.
.venv-npu/bin/python -m src.evaluate \
  --dataset SPEED --dataset_root_dir /data/datasets/SPEED_DUAL_TOKEN_SMOKE \
  --vit_model vit_b_16_no_weights --pose_decoder geometry_dual_token \
  --rotation_format matrix --merge_outputs --no_mlp --dropout_rate 0 \
  --model_weights mig_docs/working/dual_token/cli_finetuned.pth \
  --batch_size 4 --num_workers 0 --skip_tensorboard \
  --log_dir mig_docs/working/dual_token/eval_logs \
  --output_json mig_docs/working/dual_token/cli_predictions.json
```

`src.predict` accepts the same decoder flags plus `--use_test_split` and
`--output_csv`. To evaluate the baseline, omit the decoder flag and load
`weights/vit_b_16.pth`. Always match decoder width/heads/layers/interaction options to
the checkpoint; explicit loading rejects incompatible metadata or state keys.

## Validation evidence

Environment gate AUTO: Ascend 910B2C, existing CANN 9.0.0, torch 2.10.0+cpu,
torch_npu 2.10.0, `.venv-npu`, device `npu:0`. No environment changes were required.

`scripts/validate_geometry_decoder.py` passed:

- Baseline output regression: exact match with saved pre-edit outputs.
- Four real training images: NPU/CPU FP32 maximum absolute difference 0.000144005;
  NPU AMP FP16/CPU FP32 difference 0.000328183. Tolerances are 0.005/0.005 and
  0.01/0.01 respectively (atol/rtol).
- Independent image/geometry perturbations both change the output.
- Forty decoder-pretraining steps and eight joint-training steps with finite gradients.
  Geometry encoder, both task queries/attentions and both gates receive nonzero gradients.
  Frozen backbone parameters stay unchanged; unfrozen backbone parameters update.
- Fixed four-image batch FP32 loss before/after optimization: 2.309073 -> 0.122014
  (94.7% reduction). This is a tiny-batch overfit check, not generalization performance.
- Checkpoint metadata reconstruction/save/load: exact NPU output agreement.
- Reconstructed rotation matrices are orthogonal within 1e-4.
- 384-resolution AMP inference passes on a resized real image; no full 384 training run.
- No-interaction ablation CPU backward passes.

The standard `src.train` CLI also completed one frozen and one unfrozen epoch on an
isolated 32-train/8-val/8-test view of SPEED_FIXED. The joint stage used the normal
data-augmentation path, exercising geometry after augmentation. Both stages saved
checkpoints. The view contains symlinks to existing images and separate truncated
annotation JSON files; original datasets are unchanged.

Raw outputs and logs: `mig_docs/working/dual_token/`. The tiny-batch checkpoint is
`geometry_dual_token_smoke.pth`; CLI stages produce `cli_pretrained.pth` and
`cli_finetuned.pth`. These are functional-test artifacts, not final trained weights.

Limitations: full pretraining/convergence, independent test accuracy, improvement
over baseline, detector-to-pose chained accuracy, SPEED+ and multi-card training have
not been validated. ViT initialization comes from the supplied pose checkpoint;
its training overlap with this dataset is unknown. No GPU performance comparison.

## Source changes and rollback

- `src/geometry_decoder.py`: geometry encoding, task decoder, integration helper/CLI options.
- `src/vit_models.py`: spatial-feature method; existing forward unchanged.
- `src/models.py`: opt-in decoder, freeze controls, explicit backbone initialization,
  metadata checkpoints and reconstruction.
- `src/engine.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`: geometry passed
  through the real data path; baseline calls unchanged. Zero-worker training data
  loaders now omit prefetching to support the smoke run.
- `scripts/validate_geometry_decoder.py`, `scripts/pretrain_geometry_decoder.sh`:
  reproducible functional validation and full-data two-stage recipe.

Pre-change source archive: `/data/FastPose-ViT-before-dual-token-20261107-020201.tar.gz`.
Original dataset and weight files are unchanged. Work remains uncommitted.

## Standard CLI inference results

All five invocations succeeded: frozen pretraining, joint training with augmentation,
new-model evaluation, new-model CSV prediction, and baseline evaluation. Evaluation
JSON files contain eight finite pose predictions each; CSV has eight rows. On this
tiny test view the short-trained new model SPEED score is 2.6228, versus 0.0094 for
the supplied baseline checkpoint. This is not a matched-training comparison and
does not demonstrate an accuracy gain. The new weights require substantive training.
Validation loss after the frozen/joint CLI stages was 2.2231/2.2687 respectively.

Current process snapshot: validation jobs completed; no full pretraining run launched.
