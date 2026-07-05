# Model Card

## Model summary

This repository is a conservative research scaffold for proxy motor-behavior recognition on a public SSBD-style stereotyped motor behavior task. The implemented workflows are centered on numeric pose keypoints, windowed feature engineering, lightweight baseline classifiers, and reproducible evaluation scaffolding.

The current repository does not ship a trained deployment model or a clinical decision system. It does include a completed accessible-video benchmark report with conservative, research-only metrics.

## Intended use

- Proxy motor-behavior recognition research scaffold
- Public SSBD-style stereotyped motor behavior task exploration
- Numeric pose/keypoint and feature-based experimentation
- Reproducible evaluation and provenance workflows for future allowed-data runs
- Research-only benchmark reporting on accessible public video metadata

## Out-of-scope use

- This repository is not an autism diagnosis tool, not an autism screening tool, not clinical triage, not surveillance, and not deployment-ready
- Autism diagnosis
- Autism screening
- Clinical triage
- Surveillance
- Deployment-ready decision support
- Replacement for professional assessment
- Clinical diagnosis or screening
- Public-health or medical-device claims

## Benchmark results

These results are from the completed accessible-video benchmark run documented in `docs/full_ssbdplus_benchmark_report.md`.

Dataset snapshot for the final Stage G comparison:

- Accessible SSBD+ videos processed: 28
- Feature windows: 1,178
- Positive windows: 349
- Negative windows: 829
- Final Stage G feature count: 48 total features, including 42 `ms_*` multi-scale features

| Method | Distillation | Multi-scale | GroupKFold AUROC | GroupKFold AUPRC | LOSO AUROC | LOSO AUPRC | Permutation p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Logistic Baseline | No | No | 0.659 | 0.440 | 0.665 | 0.596 | — |
| Current Random Forest | No | No | 0.604 | 0.387 | 0.575 | 0.531 | — |
| MS-STF only | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| Teacher only | No | Yes | 0.637 | 0.449 | 0.607 | 0.606 | — |
| Student hard | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| D-MS-STF proposed | Yes | Yes | 0.625 | 0.447 | 0.633 | 0.590 | 0.000999 |

Calibration snapshot:

| Protocol | Model | Brier | ECE |
| --- | --- | ---: | ---: |
| GroupKFold | Logistic Regression | 0.232 | 0.197 |
| GroupKFold | Teacher only | 0.208 | 0.139 |
| GroupKFold | D-MS-STF proposed | 0.223 | 0.166 |
| LOSO | Logistic Regression | 0.238 | 0.286 |
| LOSO | Teacher only | 0.233 | 0.287 |
| LOSO | D-MS-STF proposed | 0.244 | 0.279 |

Permutation test summary: D-MS-STF GroupKFold AUROC, 1,000 permutations,
p-value 0.000999.

Final interpretation: Stage G completed a real D-MS-STF run on the
accessible-video SSBD+ benchmark. D-MS-STF was statistically above the
within-group permutation null, did not outperform the logistic baseline on
AUROC, provided a small GroupKFold AUPRC improvement, and showed somewhat
better GroupKFold calibration than logistic. Teacher-only achieved better
calibration/Brier in some settings. The proposed method is a tested research
contribution and ablation framework, not a demonstrated superior or SOTA model.

## Training and evaluation status

- Completed benchmark reporting is available for accessible public videos only
- No clinical validation has been performed
- No deployed model is provided
- The codebase currently provides scaffolding, synthetic tests, and reproducible evaluation structure in addition to the completed benchmark report
- Stage G results are mixed and do not support an overall superiority claim for D-MS-STF over the logistic baseline

## Data and privacy

- Raw videos, frames, and images are not committed to this repository
- Numeric keypoints and derived numeric features are the intended persistence targets
- Public dataset access must follow source terms, usage restrictions, and this repository's ethics policy
- Privacy-safe visual outputs, if generated externally for review, should remain abstract skeleton renderings rather than identifiable child imagery
- Generated Stage G artifacts such as `aggregate_metrics.csv`, `fold_metrics.csv`, `report.json`, `features_with_ms.csv`, and `ssbd_stage_g_final_results.zip` should stay out of version control unless explicitly intended

See also: `docs/data_ethics_policy.md`.

## Validation plan for future allowed-data runs

- Group-disjoint baselines to reduce leakage
- Leave-one-subject or leave-one-group-out validation
- Within-group permutation testing
- Provenance manifests for numeric inputs and downstream artifacts
- SVG validation for any generated figures

## Limitations

- Public dataset limitations may constrain usable sample count and coverage
- Small-data settings increase instability and overfitting risk
- Pose-estimation artifacts may distort downstream numeric features
- Demographic, environmental, and recording-context bias risks are unresolved
- Spurious correlations may dominate learned behavior labels
- No diagnostic validity is established
- No screening validity is established
- No clinical validation is established

## Safety statement

Any outputs from this repository must be interpreted as exploratory motor-behavior classification signals only. They are not diagnostic evidence, not screening evidence, not a basis for clinical decision-making, and not deployment-ready.
