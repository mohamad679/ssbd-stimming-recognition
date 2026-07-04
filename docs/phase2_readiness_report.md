# Phase 2 Readiness Report

This project remains a proxy motor-behavior recognition effort built on public SSBD data. It is not an autism diagnostic or screening tool, has not been clinically validated, and must never be described as one.

## Phase 2 completed

Phase 2 is complete and ready to close.

- `scripts/run_baselines.py` now runs deterministic baseline classifiers: class-weighted logistic regression and random forest.
- Validation is group/video-disjoint via deterministic `GroupKFold`, grouped by video or child identifier.
- The honest runner reports binary metrics for each fold: AUROC, AUPRC, Brier score, and ECE.
- Fold-level metrics are printed explicitly, along with per-model summaries.
- Metric summaries handle unavailable AUROC/AUPRC honestly: when a fold has only one class, discrimination metrics are tracked as unavailable instead of being imputed.
- The isolated leaky-split ablation exists under `scripts/diagnostics/leaky_split_ablation.py` as a diagnostic only.

## Honest validation status

The main baseline path is the defended benchmark.

- `scripts/run_baselines.py` uses group-disjoint `GroupKFold`.
- There is no leaky flag in the honest runner.
- Train and test groups/video IDs must not overlap.
- Model artifacts are not persisted by the runner.

## Leaky diagnostic status

The leakage ablation is intentionally separate from the honest benchmark.

- `scripts/diagnostics/leaky_split_ablation.py` exists only as a diagnostic.
- It prints a loud warning that the results are leaky and must not be reported as valid performance.
- Its purpose is leakage sensitivity comparison only.
- It is not imported by or wired into `scripts/run_baselines.py`.

## Safety and privacy status

The repository remains constrained to numeric, non-identifiable data products.

- No raw video is committed.
- No frames, images, or face crops are committed.
- No SSBD XML or CSV source data is copied into the repo.
- No model artifacts are committed.
- No result artifacts are committed.
- Modeling operates only on numeric feature tables.

## Current test status

Latest verification:

```text
python3 -m pytest
```

Exact result: **70 passed in 18.65s**.

Pytest also emitted a `pytest-asyncio` `PytestDeprecationWarning` because `asyncio_default_fixture_loop_scope` is unset. That warning is environment-level and non-blocking.

## Phase 3 readiness

The repository is ready for Phase 3 work, with the following additions still pending:

- Leave-One-Child-Out validation can be added next.
- Permutation testing can be added next, and it must use the exact same model configuration as the defended benchmark.
- An artifact provenance manifest should be added in Phase 3.
- SVG/XML figure validation should be added later before committing generated figures.
- No diagnostic/autism screening claims are permitted.

## Known not-yet-done items

- No real MediaPipe extraction run has been performed.
- No video availability verification has been performed.
- No real benchmark results have been produced.
- No LOSO evaluation has been implemented.
- No permutation test has been implemented.
- No artifact manifest has been implemented.
- No clinical validation has been performed.

## Scope note

This report records readiness for the next phase only. It does not add Phase 3 functionality.
