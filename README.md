This is a proxy motor-behavior recognition task (arm-flapping / head-banging / spinning vs. no stereotypy) built on a public research dataset. It is not an autism diagnostic or screening tool, has not been clinically validated, and must never be described as one.

# SSBD self-stimulatory behavior recognition

Lightweight research repository focused on proxy motor-behavior recognition from public dataset video.

## Current status

- Phase 0: ethics policy, SSBD+ metadata parsing, and metadata-only access reporting; no video downloads or availability verification
- Phase 1: numeric pose-keypoint model and CSV I/O, video-scoped labeled windows, interpretable features, a safe MediaPipe extraction scaffold, and a synthetic feature-table pipeline
- Phase 2: deterministic logistic-regression and random-forest baselines, group/video-disjoint `GroupKFold`, fold-level AUROC/AUPRC/Brier/ECE reporting, and an isolated leaky-split diagnostic
- Phase 3: complete as a scaffold, with LOSO validation, within-group permutation testing, artifact provenance manifest tooling, and SVG/XML validation tooling available
- Phase 4: complete as a scaffold, with privacy-safe skeleton SVG visualization and model-native numeric feature-importance tooling
- Phase 5: complete as a scaffold, with packaging metadata, conservative model-card and limitations documentation, and GitHub Actions CI for lightweight synthetic checks
- Current benchmark status: completed accessible-video Colab run documented in `docs/full_ssbdplus_benchmark_report.md`
- Stage F: D-MS-STF adds multi-scale numeric skeleton features and strictly fold-local, inner-cross-fitted teacher/student evaluation; see `docs/stage_f_distilled_ms_stf.md`
- Stage G: a numeric-only Colab workflow reuses existing SSBD+ keypoint/feature artifacts, runs GroupKFold and LOSO D-MS-STF, and creates privacy-safe reports; see `docs/stage_g_d_ms_stf_colab.md`
- Benchmarked on 36 SSBD+ metadata videos; 28 were successfully processed and 8 were unavailable at access time
- Feature table size: 65 segments, 1,178 windows, 349 positive windows, 829 negative windows, about 29.6% positive prevalence
- Safe artifacts only: final outputs are numeric CSV/JSON/TXT/SVG reports; raw videos, frames, images, MediaPipe `.task` files, trained model binaries, and the result zip stay out of version control
- Available scripts: `scripts/run_loso.py`, `scripts/run_permutation_test.py`, `scripts/run_distilled_ms_stf.py`, `scripts/benchmark/run_stage_g_d_ms_stf_colab.py`, `scripts/build_artifact_manifest.py`, `scripts/validate_svgs.py`
- Visualization and interpretation: `scripts/render_skeleton_svg.py` renders abstract stick-figure SVGs from numeric keypoint CSVs, and `scripts/explain_feature_importance.py` summarizes exploratory, non-causal model-native importances from numeric feature tables; neither reads raw frames, images, or videos
- Documentation: see `docs/full_ssbdplus_benchmark_report.md`, `docs/model_card.md`, `docs/limitations.md`, and `docs/final_project_status.md`
- CI: GitHub Actions runs `pytest` and `compileall` against synthetic, privacy-safe inputs only; it does not download videos or run real SSBD inference

| Protocol | Model | AUROC | AUPRC | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| GroupKFold, 5 folds | Logistic Regression | 0.659 ± 0.024 | 0.440 ± 0.137 | 0.232 | 0.197 |
| GroupKFold, 5 folds | Random Forest | 0.591 ± 0.044 | 0.368 ± 0.106 | 0.220 | 0.151 |
| LOSO, 28 folds | Logistic Regression | 0.665 ± 0.179 | 0.596 ± 0.296 | — | — |
| LOSO, 28 folds | Random Forest | 0.571 ± 0.165 | 0.528 ± 0.280 | — | — |

Planned top-level layout:

- `src/ssbd_behavior/` Python package
- `tests/` minimal smoke tests
- `scripts/` future utility entry points
- `configs/` future experiment and runtime configuration
- `docs/` supporting documentation

Data and generated artifacts are intentionally excluded from version control.
