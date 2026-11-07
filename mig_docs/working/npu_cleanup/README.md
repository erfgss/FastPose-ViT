# NPU-only code cleanup — 2026-11-07
Removed TensorRT conversion/evaluation, NVIDIA monitoring, NVIDIA Dockerfile, compiled deformable-attention extension and build scripts. Removed CUDA training/stream/timing branches and NVIDIA dependency packages/index. Device selection now accepts NPU and CPU only. ONNX export remains portable; ONNX Runtime uses CPU. Distributed helper uses HCCL; multi-card execution remains untested.

Verification: Python syntax compilation passed. Real SPEED images: 32-image CPU/NPU comparison passed with FP32 max absolute difference 0.000463009 and FP16 0.000968933. Three FP16 real-data training steps passed with finite gradients. Portable attention CPU/NPU forward/backward and detector FP32 backward passed. See real_data.log and attention.log.

ONNX export was not runtime-tested: optional onnx package is absent in the current environment. No optional packages were installed as part of this cleanup. Historical migration reports retain the earlier backend comparisons. Dataset and checkpoint files are unchanged.

Backups: /data/FastPose-ViT-before-npu-cleanup-20261107-014101.tar.gz and /data/FastPose-ViT-cleanup-config-backup.tar.gz. Changes are not committed.
