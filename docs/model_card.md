# Model Card

## Model summary

This repository is a conservative research scaffold for proxy motor-behavior recognition on a public SSBD-style stereotyped motor behavior task. The implemented workflows are centered on numeric pose keypoints, windowed feature engineering, lightweight baseline classifiers, and reproducible evaluation scaffolding.

The current repository does not ship a trained deployment model, real SSBD benchmark outputs, or real full-video inference results.

## Intended use

- Proxy motor-behavior recognition research scaffold
- Public SSBD-style stereotyped motor behavior task exploration
- Numeric pose/keypoint and feature-based experimentation
- Reproducible evaluation and provenance workflows for future allowed-data runs

## Out-of-scope use

- This repository is not an autism diagnosis tool, not an autism screening tool, not clinical triage, not surveillance, and not deployment-ready
- Autism diagnosis
- Autism screening
- Clinical triage
- Surveillance
- Deployment-ready decision support
- Replacement for professional assessment

## Training and evaluation status

- No real SSBD benchmark metrics are claimed in this repository
- No real full-video inference is claimed in this repository
- No clinical validation has been performed
- No deployed model is provided
- The codebase currently provides scaffolding, synthetic tests, and reproducible evaluation structure only

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

## Safety statement

Any outputs from this repository must be interpreted as exploratory motor-behavior classification signals only. They are not diagnostic evidence, not screening evidence, and not a basis for clinical decision-making.
