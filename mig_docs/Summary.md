> Final SPEED test evaluation (2026-11-07): full training completed; selected joint epoch 30. On 1200 matched test images, baseline SPEED score 0.0099751 versus dual-token 0.0124502 (error +24.81%). No accuracy gain in this checkpoint comparison. Full results: experiments/geometry_dual_token_full_20261107_022724_UTC/speed_test_comparison/README.md. Older in-progress and not-yet-tested statements below are historical.

> Cleanup 20261107_034803: completed smoke weights, CLI prediction outputs and temporary logs were removed with user confirmation. Summary metrics and baseline_before.pt regression reference remain. Current full-training logs and best weights are retained. Historical paths to removed smoke artifacts are no longer live.

> SPEED+ data update (2026-11-07): prepared /data/datasets/SPEED_PLUS_FIXED (47966 train / 11994 val / 9531 real test); CRC, image headers, bbox coverage and sampled loader/CPU forward checks passed. See working/SPEED_PLUS_preparation.md. Historical statements below that SPEED+ images are missing are superseded. No SPEED+ accuracy benchmark has been run.

> Full training launched on 2026-11-07 (UTC): 10 frozen-backbone pretraining epochs followed by 30 joint epochs, full SPEED_FIXED 9600/1200 split. Active experiment: experiments/geometry_dual_token_full_20261107_022724_UTC. Logs: pretrain.log / finetune.log; best weights: pretrained.pth / finetuned.pth. See status.txt and run.json for current run. Earlier statements that full training was not launched describe the previous validation stage.

# Current extension: geometric dual-token decoder

2026-11-07: implemented opt-in geometry-conditioned translation/rotation queries in
FastPoseViT; default baseline remains unchanged (pre/post NPU output max difference 0).
Environment AUTO: Ascend 910B2C, existing torch/torch_npu 2.10.0 and CANN 9.0.0.
CPU golden source is the same new model on real SPEED images; no GPU baseline.
Forty frozen-backbone pretraining steps + eight joint steps passed on a fixed
four-image batch; FP32 loss 2.309073 -> 0.122014. NPU/CPU FP32 max difference
0.000144005; AMP FP16 max difference 0.000328183. Save/reload exactly reproduces outputs.
384 AMP inference passed. Standard CLI frozen/joint training, eval and predict also passed
on a 32/8/8 smoke split. New-model smoke test SPEED score 2.6228 versus baseline 0.0094:
short-trained new weights are NOT an improved model. Full pretraining and independent
generalization evaluation remain outstanding; SPEED+ is unavailable.
Current processes: validation jobs completed; no full pretraining job launched.
Report: working/Geometry_dual_token.md. Logs/checkpoints: working/dual_token/.
Source backup: /data/FastPose-ViT-before-dual-token-20261107-020201.tar.gz.

---
Previous baseline validation follows (different model/training scope):

# Ascend migration summary

Updated 2026-11-07: SPEED real-data validation completed on Ascend 910B2C. Both pose checkpoints and the detector passed evaluation on 1200 internally split images. Real-image CPU/NPU numerical checks and 3 FP16 training steps passed.

- Pose 224: translation 0.0231 m; rotation 0.0078 rad; SPEED score 0.0100.
- Pose 384: translation 0.0221 m; rotation 0.0074 rad; SPEED score 0.0093.
- Detector small: repository AP@0.50 0.9992; AP@[0.50:0.95] 0.9519.
- Pose results use annotation boxes; detector evaluated separately.
- This is an internal split of original labeled training data; checkpoint overlap is unknown. Not an official blind-test result or convergence verification.
- SPEED+ images are absent and remain unvalidated.

Current authoritative report and reproduction commands: working/SPEED_validation.md. Evidence: working/speed_validation/. Earlier migration reports describe the initial synthetic stage; their missing-SPEED statements are superseded by this report. Code changes remain uncommitted.


Residual v2 optimization completed: 8 frozen-branch epochs on NPU, best epoch6 by validation. Internal SPEED score baseline0.00997514 / v1joint0.01245017 / v2residual0.00995603. Tiny gain, not proof of stable superiority. Details: experiments/geometry_residual_v2/run01/README.md


## Literature-guided experiments, 2026-11-07
Completed 15 primary +3 supplementary frozen-residual training runs (8epochs each), 3seeds; 5 SPEED robustness conditions and full9531 SPEED+ real cross-domain evaluation. Validation-selected disentangled translation loss, fixed seed123. Standard SPEED holdout score baseline0.0099751437 -> selected0.0099523757 (~0.228%); rotation unchanged. NPU batch1 network median3.27ms ->4.45ms. Large bbox shifts and SPEED+ domain gap remain unresolved (~65.36deg mean rotation on real SPEED+). Not official blind-test or convergence evidence. Complete report: experiments/literature_suite_20261107/run02/REPORT.md; selected_model.pth and all decoder deltas retained.
