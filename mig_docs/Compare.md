

Residual v2 optimization completed: 8 frozen-branch epochs on NPU, best epoch6 by validation. Internal SPEED score baseline0.00997514 / v1joint0.01245017 / v2residual0.00995603. Tiny gain, not proof of stable superiority. Details: experiments/geometry_residual_v2/run01/README.md


## Literature-guided experiments, 2026-11-07
Completed 15 primary +3 supplementary frozen-residual training runs (8epochs each), 3seeds; 5 SPEED robustness conditions and full9531 SPEED+ real cross-domain evaluation. Validation-selected disentangled translation loss, fixed seed123. Standard SPEED holdout score baseline0.0099751437 -> selected0.0099523757 (~0.228%); rotation unchanged. NPU batch1 network median3.27ms ->4.45ms. Large bbox shifts and SPEED+ domain gap remain unresolved (~65.36deg mean rotation on real SPEED+). Not official blind-test or convergence evidence. Complete report: experiments/literature_suite_20261107/run02/REPORT.md; selected_model.pth and all decoder deltas retained.
