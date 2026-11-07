# Environment — 2026-11-07

Gate: AUTO. Remote npu-smi reports healthy Ascend 910B2C 64GB, driver 26.0.rc1; existing CANN 9.0.0 environment retained. .venv-npu: Python 3.11, torch 2.10.0+cpu, torch_npu 2.10.0; torch.npu.is_available()=True. README and requirements-pytorch.txt match the installed NPU/CPU framework. No framework/system changes needed.

SPEED is available at /data/datasets/SPEED_FIXED. SPEED+ images remain unavailable. Current work adds an optional geometric dual-token decoder to the already adapted PyTorch code; baseline behavior is retained. FP32 inference and AMP FP16 training will be validated on npu:0. No GPU baseline is available.
