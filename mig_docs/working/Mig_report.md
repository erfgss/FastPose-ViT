> Cleanup 20261107_034803: completed smoke weights, CLI prediction outputs and temporary logs were removed with user confirmation. Summary metrics and baseline_before.pt regression reference remain. Current full-training logs and best weights are retained. Historical paths to removed smoke artifacts are no longer live.

> Current extension (2026-11-07): optional geometry dual-token decoder, frozen pretraining and joint training passed on NPU. See [Geometry_dual_token.md](Geometry_dual_token.md) for exact changes, evidence, commands and limitations. Baseline retained; no demonstrated accuracy improvement.

> Update 2026-11-07: real SPEED validation is now complete; see [SPEED_validation.md](SPEED_validation.md). Any missing-SPEED or synthetic-only status below describes the earlier migration stage. SPEED+ remains unavailable.

# Migration report

## 1 Scope
Single-card PyTorch training/inference adaptation of pose estimator and LW-DETR. FP32 first, FP16 pose smoke.

## 2.2 Feasibility
Route: torch_npu explicit device and AMP; detector replaces CUDA-only MSDA with repository-native differentiable grid_sample implementation on NPU. Pre/postprocessing remains original Python/CPU code. Contract: NCHW float input; pose outputs translation 3 plus rotation 6 merged; detector boxes/logits.
High risk: CUDA extension (replaced, forward/backward compared); medium: AMP overflow (scale reduced), CUDA compile (disabled on NPU); full data and business metrics unavailable.
Phase 1 checkpoint forward/backward -> Phase 2 CPU golden and 224/384 pose plus 640 detector -> Phase 3 real dataset evaluation/target batches and optional optimizations, pending data.

## 3 Environment
Validated on 2026-11-06: Ascend 910B2C (64GB), driver 26.0.rc1, CANN 9.0.0, Python 3.11.15, torch 2.10.0+cpu, torch_npu 2.10.0, torchvision 0.25.0+cpu. Environment gate: AUTO. Isolated .venv-npu; system framework preserved.

## 4 Changes
src/device.py adds explicit backend selection, synchronization and NPU scaler. Pose/detector train/eval/predict entrypoints select NPU automatically. Engines autocast using actual device. CUDA-specific compilation and tuning remain gated to CUDA. Latency synchronization covers NPU. Custom ViT backbone weights load onto CPU first. CUDA MSDA import is optional and CPU/NPU use PyTorch attention. requirements-npu.txt avoids CUDA/TensorRT packages.

## 5 Reproduction
See Mig_Readme.md; all edits are uncommitted for review. .venv-npu is ignored. Existing model weights are unchanged. Roll back tracked changes with a reviewed git diff, and remove new NPU-only files if desired; do not remove checkpoints.

## 6 Validation
PASS: CLI help for pose train/evaluate/predict and detector train; compileall and git diff --check; both pretrained pose models CPU/NPU golden and FP16 finite forward; ViT FP16 backward/AdamW step; real detector checkpoint forward/backward; MSDA forward/backward CPU agreement. See Compare.md.
NOT RUN: full SPEED evaluation, real training short-test, multi-card, detector AMP, complete data preprocessing integration.

## 7 Failures and fixes
Initial torchvision PyPI wheel could not load its CUDA libraries (torchvision::nms missing). Replaced with matching 0.25.0+cpu wheel in isolated venv; imports/CLI then pass.
Initial synthetic FP16 train step overflowed at default scale 65536. Reduced NPU scale to 256; all model gradients and optimizer step verified finite. src/train.py uses this NPU initial scale; scaler retains dynamic adjustment.
System ms-service-profiler has pre-existing unrelated missing dependency constraints. No system package replacements performed.

## 8 Evidence
mig_docs/working/smoke.log, smoke_results.json, attention.log; scripts/smoke_npu.py and scripts/check_npu_attention.py are reproducible checks. No original project baseline training logs were found.

## Full training startup correction (2026-11-07)
Initial full-run launch stopped before the first optimizer step because active-parameter logging referenced undefined format_number. Replaced it with integer logging; failure evidence retained under experiments/geometry_dual_token_full_20261107_022724_UTC/attempt1_failed. Restarted with gradient clipping max_norm=1.0, matching the earlier functional validation.
