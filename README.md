This is a proxy motor-behavior recognition task (arm-flapping / head-banging / spinning vs. no stereotypy) built on a public research dataset. It is not an autism diagnostic or screening tool, has not been clinically validated, and must never be described as one.

# SSBD self-stimulatory behavior recognition

Lightweight research repository focused on proxy motor-behavior recognition from public dataset video.

## Current status

- Phase 0: ethics policy, SSBD+ metadata parsing, and metadata-only access reporting; no video downloads or availability verification
- Phase 1: numeric pose-keypoint model and CSV I/O, video-scoped labeled windows, interpretable features, a safe MediaPipe extraction scaffold, and a synthetic feature-table pipeline
- Phase 2: deterministic logistic-regression and random-forest baselines, group/video-disjoint `GroupKFold`, fold-level AUROC/AUPRC/Brier/ECE reporting, and an isolated leaky-split diagnostic
- Phase 3: complete as a scaffold, with LOSO validation, within-group permutation testing, artifact provenance manifest tooling, and SVG/XML validation tooling available
- Available Phase 3 scripts: `scripts/run_loso.py`, `scripts/run_permutation_test.py`, `scripts/build_artifact_manifest.py`, `scripts/validate_svgs.py`
- Full verification: 70 tests passed; the environment-level `pytest-asyncio` warning is non-blocking
- Next: Phase 4 interpretability skeleton visualizations
- Not yet implemented: real MediaPipe extraction, real benchmark results, real SSBD video inference, interpretability figures, packaging/model card, or clinical validation

Planned top-level layout:

- `src/ssbd_behavior/` Python package
- `tests/` minimal smoke tests
- `scripts/` future utility entry points
- `configs/` future experiment and runtime configuration
- `docs/` supporting documentation

Data and generated artifacts are intentionally excluded from version control.
