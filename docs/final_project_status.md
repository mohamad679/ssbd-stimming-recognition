# Final Project Status

## Completed phase summary

- Phase 0: data access and ethics scaffold completed with dataset-access documentation and repository guardrails
- Phase 1: pose/keypoint and feature pipeline scaffold completed for numeric processing only
- Phase 2: baseline and group-disjoint validation scaffold completed
- Phase 3: LOSO, permutation, provenance, and SVG validation scaffold completed
- Phase 4: skeleton SVG and feature-importance interpretability scaffold completed
- Phase 5: packaging, model card, limitations, and CI scaffold completed
- Completed accessible-video SSBD+ benchmark run documented in `docs/full_ssbdplus_benchmark_report.md`
- Stage G completed a real accessible-video Distilled Multi-Scale Skeleton Temporal Feature Learning (D-MS-STF) benchmark run with GroupKFold, LOSO, and 1,000 within-group permutations
- Final Stage G cohort summary: 28 processed videos, 1,178 windows, 349 positive windows, 829 negative windows, and 48 total features after augmentation including 42 `ms_*` features

## Current limitations

- The benchmark run is research-only, accessible-video only, and not a clinical or screening validation
- No clinical validation has been performed
- No deployment system is provided
- The repository does not claim diagnostic, screening, SOTA, or production readiness
- The final Stage G results are mixed: D-MS-STF is statistically above the within-group permutation null, but it does not outperform the logistic baseline on AUROC
- Teacher-only achieves better calibration/Brier in some settings, so the repository does not present D-MS-STF as the overall best model

## Safe next steps

- Preserve the numeric/report artifact boundary
- Re-run the benchmark only on permitted data and only with privacy-safe outputs
- Record provenance for any future numeric inputs and reports
- Review any generated artifacts before deciding whether they belong in version control
- Do not commit raw videos, frames, images, MediaPipe `.task` files, model binaries, `features_with_ms.csv`, `fold_metrics.csv`, `aggregate_metrics.csv`, `report.json`, or `ssbd_stage_g_final_results.zip` unless explicitly intended

## Repository status statement

The repository is complete as a non-diagnostic research scaffold with a
completed accessible-video benchmark report and a completed Stage G D-MS-STF
research evaluation. It is suitable for reproducible packaging, documentation,
CI, and future privacy-conscious numeric experimentation. The final empirical
conclusion is mixed: D-MS-STF is a valid proposed method and ablation
framework, but it is not demonstrated here as a superior model on this small
accessible-video cohort and does not justify diagnostic, clinical,
deployment, screening, SOTA, or production claims.
