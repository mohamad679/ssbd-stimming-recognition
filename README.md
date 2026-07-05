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
- Stage F: Distilled Multi-Scale Skeleton Temporal Feature Learning (D-MS-STF) adds multi-scale numeric skeleton features and strictly fold-local, inner-cross-fitted teacher/student evaluation; see `docs/stage_f_distilled_ms_stf.md`
- Stage G: completed a real accessible-video SSBD+ Distilled Multi-Scale Skeleton Temporal Feature Learning (D-MS-STF) benchmark run with GroupKFold, LOSO, and 1,000 within-group permutations; see `docs/stage_g_d_ms_stf_colab.md`
- Benchmarked on 36 SSBD+ metadata videos; 28 were successfully processed and 8 were unavailable at access time
- Feature table size: 65 segments, 1,178 windows, 349 positive windows, 829 negative windows, about 29.6% positive prevalence
- Stage G augmented feature set: 48 total features, including 42 `ms_*` multi-scale features
- Safe artifacts only: final outputs are numeric CSV/JSON/TXT/SVG reports; raw videos, frames, images, MediaPipe `.task` files, trained model binaries, and result zips stay out of version control
- Available scripts: `scripts/run_loso.py`, `scripts/run_permutation_test.py`, `scripts/run_distilled_ms_stf.py`, `scripts/benchmark/run_stage_g_d_ms_stf_colab.py`, `scripts/build_artifact_manifest.py`, `scripts/validate_svgs.py`
- Visualization and interpretation: `scripts/render_skeleton_svg.py` renders abstract stick-figure SVGs from numeric keypoint CSVs, and `scripts/explain_feature_importance.py` summarizes exploratory, non-causal model-native importances from numeric feature tables; neither reads raw frames, images, or videos
- Documentation: see `docs/full_ssbdplus_benchmark_report.md`, `docs/model_card.md`, `docs/limitations.md`, and `docs/final_project_status.md`
- CI: GitHub Actions runs `pytest` and `compileall` against synthetic, privacy-safe inputs only; it does not download videos or run real SSBD inference

Final Stage G comparison on the accessible-video SSBD+ cohort:

| Method | Distillation | Multi-scale | GroupKFold AUROC | GroupKFold AUPRC | LOSO AUROC | LOSO AUPRC | Permutation p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Logistic Baseline | No | No | 0.659 | 0.440 | 0.665 | 0.596 | — |
| Current Random Forest | No | No | 0.604 | 0.387 | 0.575 | 0.531 | — |
| MS-STF only | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| Teacher only | No | Yes | 0.637 | 0.449 | 0.607 | 0.606 | — |
| Student hard | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| D-MS-STF proposed | Yes | Yes | 0.625 | 0.447 | 0.633 | 0.590 | 0.000999 |

Calibration snapshot for the main research comparison:

| Protocol | Model | Brier | ECE |
| --- | --- | ---: | ---: |
| GroupKFold | Logistic Regression | 0.232 | 0.197 |
| GroupKFold | Teacher only | 0.208 | 0.139 |
| GroupKFold | D-MS-STF proposed | 0.223 | 0.166 |
| LOSO | Logistic Regression | 0.238 | 0.286 |
| LOSO | Teacher only | 0.233 | 0.287 |
| LOSO | D-MS-STF proposed | 0.244 | 0.279 |

Empirical conclusion: mixed results. Stage G completed a real D-MS-STF run on
the accessible-video SSBD+ benchmark. D-MS-STF was statistically above the
within-group permutation null, did not outperform the current logistic baseline
on AUROC, provided a small GroupKFold AUPRC gain, and showed somewhat better
GroupKFold calibration than logistic. Teacher-only achieved better
calibration/Brier in some settings. The proposed method is documented as a
tested research contribution and ablation framework, not as a superior or SOTA
model.

The Colab run produced generated artifacts such as `aggregate_metrics.csv`,
`fold_metrics.csv`, `report.json`, `features_with_ms.csv`, and
`ssbd_stage_g_final_results.zip`. These are generated outputs and should not be
committed unless explicitly intended.

Planned top-level layout:

- `src/ssbd_behavior/` Python package
- `tests/` minimal smoke tests
- `scripts/` future utility entry points
- `configs/` future experiment and runtime configuration
- `docs/` supporting documentation

Data and generated artifacts are intentionally excluded from version control.
