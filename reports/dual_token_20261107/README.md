# Dual-token experiments and ablation figures

See [REPORT.md](REPORT.md) for findings and limitations, and [ablation_figures/README.md](ablation_figures/README.md) for tables and figures. These are exploratory frozen-branch experiments, not a claim of converged superiority or an official blind-test result.

This directory includes small metrics, per-sample error arrays, 18 run histories, and figures. Model weights, datasets, original paper PDFs, raw logs, and Python environments are not included. Their server locations and checkpoint hashes are recorded in the report and manifests.

Rebuild the ablation figures from the repository root:

    python reports/dual_token_20261107/make_ablation_figures.py

Plotting requires Python, NumPy and Matplotlib. summarize_results.py rebuilds the main summary and plot; REPORT.md additionally includes a standard-entry-point verification appendix. Training entry points are in scripts/. Paths to the locally reviewed papers in sources.json record provenance, not runtime dependencies.
