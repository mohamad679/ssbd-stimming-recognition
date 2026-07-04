# Phase 2 notes

The valid baseline evaluation in `scripts/run_baselines.py` uses group-disjoint
`GroupKFold`, grouped by video/child identifier, to prevent windows from the same
group appearing in both training and test data.

The honest runner now reports stable fold-level metrics and per-model mean/standard
deviation summaries. Folds where AUROC or AUPRC is unavailable are explicitly
counted and are not assigned substitute values. Results and model artifacts are
not persisted by default.

An intentionally leaky window-level ablation exists only at
`scripts/diagnostics/leaky_split_ablation.py`. It ignores video/child groups and
is provided solely to measure sensitivity to leakage. Its results must never be
reported as valid model performance.
