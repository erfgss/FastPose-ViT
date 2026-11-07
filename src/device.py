"""Explicit accelerator selection for Ascend NPU and CPU."""
import os
import torch
try:
    import torch_npu  # registers the NPU backend
except ImportError:
    torch_npu = None


def get_device():
    requested = os.environ.get("FASTPOSE_DEVICE", "auto")
    if requested == "auto":
        requested = "npu:0" if torch_npu is not None and torch.npu.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("npu", "cpu"):
        raise ValueError("FASTPOSE_DEVICE must be auto, npu[:index], or cpu")
    if device.type == "npu":
        if torch_npu is None or not torch.npu.is_available():
            raise RuntimeError("FASTPOSE_DEVICE requests NPU but torch_npu/device is unavailable")
        torch.npu.set_device(device)
    return device


def synchronize(device):
    if device.type == "npu":
        getattr(torch, device.type).synchronize(device)


def make_grad_scaler(device, **kwargs):
    if device == "npu":
        kwargs.setdefault("init_scale", 256.)
        return torch.npu.amp.GradScaler(**kwargs)
    return torch.amp.GradScaler(device=device, **kwargs)
