# Technical report

[Read the report (PDF)](interventional_tgmm_report.pdf)

**Interventional Robustness of a Transformer-Based Learned GMM Solver on Held-Out Public Benchmarks**

Syed Arsalaan Nadim · MBZUAI · SDS7103 Data Analysis and Visualization · May 2026

This is the archived course-project technical report, not an arXiv submission or a peer-reviewed publication. The PDF is the original compiled version dated 2 May 2026; it has not been rewritten for this release. The adjacent LaTeX and bibliography are historical source snapshots, not a complete standalone build bundle. The PDF is the canonical readable artifact.

## Interpretation notes added for the public release

The report's conclusions are exploratory and should be read with the following qualifications:

- **Fixed training geometry is a strong baseline.** Training tasks share the same per-class means within each dataset split. A diagnostic that reuses the training means and weights, without learning a transformer, closely matches the reported no-shift Wine result and slightly exceeds the reported no-shift Digits result. This control is missing from the original report, so the original comparisons do not establish a substantial advantage over a simple learned-prior baseline.
- **Residual-generator scale metadata requires correction.** The residual sampler rescales raw residuals using already-normalized scales. The resulting residual distribution's actual component scales do not generally match the `sigma_diag` values recorded as ground truth. This affects the interpretation of scale error, aggregate parameter error, and comparisons involving assumed noise scales. The archived code and results are preserved; corrected experiments have not been run.
- **The Digits spectral baseline is inapplicable.** Its implementation explicitly requires `K <= d`, whereas the Digits experiment uses `K=10` and `d=8`. Its failure is not evidence of numerical instability or transformer superiority.
- **Uncertainty is incomplete.** The report uses one training seed and one dataset split per configuration. Its task-level standard deviations do not measure training-seed or split variability.
- **Log-likelihood is evaluated on the fitted task samples**, not on independent held-out points from each task. It should not be described as predictive likelihood.

The post-report no-shift diagnostic uses the same evaluation seeds (101, 102, 103), 100 tasks per seed, fixed training-split class means and weights, and the same known scalar noise setting used by the existing evaluation. It is a control for dataset-prior reuse, not an improved general-purpose GMM solver.

| Dataset / generator | Fixed training-geometry accuracy | Reported TGMM accuracy |
|---|---:|---:|
| Wine / residual | 0.92972 | 0.93077 |
| Wine / Gaussian-diagonal | 0.89470 | 0.89698 |
| Iris / residual | 0.73148 | 0.76609 |
| Iris / Gaussian-diagonal | 0.60818 | 0.61506 |
| Digits / residual | 0.88189 | 0.88146 |
| Digits / Gaussian-diagonal | 0.94645 | 0.94586 |

Reproduce this diagnostic with `python paper/check_report_baseline.py` after installing the project dependencies.

## Recorded experiment artifacts

The repository includes the implementation snapshot supporting the expanded report and the recorded runs below. No full retraining was performed for the public release.

| Run | Directory |
|---|---|
| Wine, 3-layer transformer | [`outputs_clean_3layer`](../outputs_clean_3layer) |
| Wine, 2-layer transformer | [`outputs_clean_2layer`](../outputs_clean_2layer) |
| Iris, 3-layer transformer | [`outputs_clean_iris_3layer`](../outputs_clean_iris_3layer) |
| Iris, 2-layer transformer | [`outputs_clean_iris_2layer`](../outputs_clean_iris_2layer) |
| Digits, 3-layer transformer | [`outputs_clean_digits`](../outputs_clean_digits) |

These contain training configurations, checkpoints, histories, and evaluation summaries. Earlier Wine overlap audits remain in `outputs_clean_suite/overlap`; the recorded Iris and Digits runs include their own audit directories. The hash audits establish only the reported exact-task overlap checks, not absence of every form of shared-prior dependence.

The transformer architecture builds on existing TGMM and Set Transformer research; this project is an empirical benchmark study, not the original TGMM method. See the report's references.
