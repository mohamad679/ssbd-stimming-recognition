# Phase 3 notes

Phase 3A adds leave-one-group-out (LOSO) validation scaffolding.

Phase 3B adds within-group permutation testing.

Phase 3C adds artifact provenance manifest tooling for explicit numeric
artifacts.

The LOSO runner in `scripts/run_loso.py` evaluates the same two baseline model
types as the Phase 2 grouped runner while keeping groups fully disjoint and
reporting unavailable folds honestly:

- training folds with only one class are marked unavailable and are not fitted
- test folds with only one class report AUROC and AUPRC as unavailable
- results and model artifacts are not persisted by default

The permutation runner uses the exact same baseline trainer functions as the
defended benchmark. It does not introduce a smaller or faster permutation-only
model configuration.

The CLI defaults to 100 permutations for local safety. The final defended run
should use 1,000 permutations.

Real Phase 3 runs should record SHA256 checksums for processed keypoint files,
feature tables, and evaluation outputs so downstream benchmark, LOSO,
permutation, and access/evaluation reports can be tied back to exact numeric
inputs.

Raw videos, extracted frames, images, and model artifacts are not normal
committed provenance targets in this repo and must not be recorded by the
artifact-manifest CLI added in this phase.

This project remains a proxy motor-behavior recognition scaffold and must not be
described as a clinical or diagnostic autism screening tool.
