> Final SPEED test evaluation (2026-11-07): full training completed; selected joint epoch 30. On 1200 matched test images, baseline SPEED score 0.0099751 versus dual-token 0.0124502 (error +24.81%). No accuracy gain in this checkpoint comparison. Full results: experiments/geometry_dual_token_full_20261107_022724_UTC/speed_test_comparison/README.md. Older in-progress and not-yet-tested statements below are historical.

> Current extension (2026-11-07): optional geometry dual-token decoder, frozen pretraining and joint training passed on NPU. See [Geometry_dual_token.md](Geometry_dual_token.md) for exact changes, evidence, commands and limitations. Baseline retained; no demonstrated accuracy improvement.

> Update 2026-11-07: real SPEED validation is now complete; see [SPEED_validation.md](SPEED_validation.md). Any missing-SPEED or synthetic-only status below describes the earlier migration stage. SPEED+ remains unavailable.

# Validation and comparison

Validated on 2026-11-06: Ascend 910B2C (64GB), driver 26.0.rc1, CANN 9.0.0, Python 3.11.15, torch 2.10.0+cpu, torch_npu 2.10.0, torchvision 0.25.0+cpu. Environment gate: AUTO. Isolated .venv-npu; system framework preserved.

Synthetic batch=1, fixed seed 42. Full checkpoint loading, CPU FP32 golden, NPU FP32 and FP16 forward. Timing: 3 warmups + 10 FP32 forwards, synchronized wall clock, model only. These short measurements are indicative, not production benchmarks.

{
  "vit_b_16": {
    "fp32_max_abs": 0.00042879581451416016,
    "fp16_max_abs": 0.0002991240471601486,
    "fp32_ms": 3.238082118332386,
    "shape": [
      1,
      9
    ],
    "train_loss": 0.41601353883743286
  },
  "vit_b_16_384": {
    "fp32_max_abs": 0.0019690990447998047,
    "fp16_max_abs": 0.0030803680419921875,
    "fp32_ms": 4.5911419205367565,
    "shape": [
      1,
      9
    ]
  },
  "detector": "finite FP32 checkpoint forward at 640x640"
}

MSDA forward and all three input gradients matched CPU within atol=rtol=1e-4. Detector FP32 backward finite (synthetic loss 72.65643). ViT FP16 optimizer step finite with init_scale=256. No SPEED dataset; task accuracy, real loss convergence and GPU performance comparison: N/A.
