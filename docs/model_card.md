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

| Protocol | Model | AUROC | AUPRC | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| GroupKFold, 5 folds | Logistic Regression | 0.659 ± 0.024 | 0.440 ± 0.137 | 0.232 | 0.197 |
| GroupKFold, 5 folds | Random Forest | 0.591 ± 0.044 | 0.368 ± 0.106 | 0.220 | 0.151 |
| LOSO, 28 folds | Logistic Regression | 0.665 ± 0.179 | 0.596 ± 0.296 | — | — |
| LOSO, 28 folds | Random Forest | 0.571 ± 0.165 | 0.528 ± 0.280 | — | — |

Permutation test summary: logistic regression AUROC, 1,000 permutations, observed AUROC 0.659495, p-value 0.000999.

## Training and evaluation status

- Completed benchmark reporting is available for accessible public videos only
- No clinical validation has been performed
- No deployed model is provided
- The codebase currently provides scaffolding, synthetic tests, and reproducible evaluation structure in addition to the completed benchmark report

## Data and privacy

- Raw videos, frames, and images are not committed to this repository
- Numeric keypoints and derived numeric features are the intended persistence targets
- Public dataset access must follow source terms, usage restrictions, and this repository's ethics policy
- Privacy-safe visual outputs, if generated externally for review, should remain abstract skeleton renderings rather than identifiable child imagery

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
