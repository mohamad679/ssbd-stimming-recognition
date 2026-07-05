# Limitations

## Technical limitations

- The repository is a scaffold around numeric pose/keypoint processing and lightweight baselines, not a finished behavior-recognition system.
- Real MediaPipe extraction and real SSBD video inference are intentionally not exercised in this repository's synthetic test flow.
- Feature engineering and baseline models may miss relevant temporal structure or overfit small datasets.
- Pose estimation errors, missing landmarks, camera motion, occlusion, compression, and annotation noise can propagate into features and predictions.

## Data limitations

- The intended task depends on a small public research dataset with uncontrolled recording conditions.
- Usable data volume may be further reduced by access attrition, annotation gaps, or exclusion decisions.
- Public internet video sources can introduce inconsistent viewpoints, lighting, frame rates, and backgrounds.
- Demographic and contextual coverage may be limited and may not support broad generalization claims.
- The completed accessible-video run processed 28 of 36 metadata videos; the unavailable videos were excluded because of YouTube access and availability problems.
- Accessibility status can change over time, so the sample may drift if videos become private, unavailable, or otherwise inaccessible later.

## Evaluation limitations

- The benchmark run is research-only and non-clinical.
- The small number of independent videos makes fold-level estimates unstable, especially under LOSO.
- The observed LOSO variance is high, so single-fold outcomes should not be overinterpreted.
- Even honest group-disjoint and LOSO validation cannot remove all bias when the dataset is small and heterogeneous.
- Permutation testing and provenance help with rigor, but they do not establish clinical validity, screening validity, or deployment readiness.

## Privacy and ethics limitations

- Working with child-behavior video data raises privacy and consent constraints that must be handled outside this repository's synthetic CI flow.
- Public dataset access must follow source terms and the repository ethics policy.
- Numeric keypoints are preferred for persistence, but privacy risk is reduced rather than eliminated.
- Raw videos, frames, images, trained model binaries, and MediaPipe `.task` files must remain temporary or excluded artifacts.

## Non-diagnostic boundary

This repository is not an autism diagnosis tool, not an autism screening tool, not a clinical triage system, and not a substitute for professional assessment. The benchmark report does not provide diagnostic validity, screening validity, or clinical validation.

## What must be done before any real-world use

- Obtain lawful and ethically appropriate access to allowed source data
- Run the external data-processing pipeline on permitted data only
- Produce and review numeric features and provenance records
- Perform real, group-disjoint evaluation on appropriate held-out subjects
- Conduct domain, bias, privacy, and safety review
- Establish application-specific validation with qualified experts

## Deployment status

The current repository is not deployment-ready. It does not provide a clinically validated model, a production system, a surveillance system, or a real-world decision-support product.
