This is a proxy motor-behavior recognition task (arm-flapping / head-banging / spinning vs. no stereotypy) built on a public research dataset. It is not an autism diagnostic or screening tool, has not been clinically validated, and must never be described as one.

# SSBD self-stimulatory behavior recognition

Lightweight research repository focused on proxy motor-behavior recognition from public dataset video.

## Current status

- Phase 0: ethics policy, SSBD+ metadata parsing, and metadata-only access reporting; no video downloads or availability verification
- Phase 1: numeric pose-keypoint model and CSV I/O, video-scoped labeled windows, interpretable features, a safe MediaPipe extraction scaffold, and a synthetic feature-table pipeline
- Phase 2: deterministic logistic-regression and random-forest baselines, group/video-disjoint `GroupKFold`, fold-level AUROC/AUPRC/Brier/ECE reporting, and an isolated leaky-split diagnostic
- Full verification: 70 tests passed; the environment-level `pytest-asyncio` warning is non-blocking
- Next: LOSO validation, permutation testing, and artifact provenance for the defended benchmark
- Not yet implemented: real MediaPipe extraction, real benchmark results, LOSO evaluation, permutation testing, artifact manifests, or clinical validation

Planned top-level layout:

- `src/ssbd_behavior/` Python package
- `tests/` minimal smoke tests
- `scripts/` future utility entry points
- `configs/` future experiment and runtime configuration
- `docs/` supporting documentation

Data and generated artifacts are intentionally excluded from version control.
