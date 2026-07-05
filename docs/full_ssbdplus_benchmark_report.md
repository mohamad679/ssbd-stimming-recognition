# Full SSBD+ Benchmark Report

## Executive summary

This report documents a completed research-only benchmark run on accessible SSBD+ metadata videos. The task is proxy motor-behavior recognition from public research video, not autism diagnosis, not autism screening, not clinical validation, and not deployment-ready decision support.

Run summary:

- SSBD+ metadata videos: 36 unique videos
- Successfully processed videos: 28
- Failed or unavailable videos: 8
- Segment rows: 65
- Feature windows: 1,178
- Positive windows: 349
- Negative windows: 829
- Positive prevalence: approximately 29.6%

The strongest model signal in this run came from wrist-distance variability and repetitive wrist/head movement features. That is consistent with a non-diagnostic motor-behavior recognition task and should not be reframed as autism detection or screening.

## Research-only scope and guardrails

- This is a research report for proxy motor-behavior recognition only.
- It is not an autism diagnosis tool.
- It is not an autism screening tool.
- It is not a clinical triage system.
- It is not clinically validated.
- It is not deployment-ready.
- It is not a medical device claim.
- It is not SOTA evidence.
- It is not intended to support real-world clinical decisions.

## Dataset and availability summary

| Item | Count | Notes |
| --- | ---: | --- |
| Unique metadata videos | 36 | SSBD+ metadata entries in the Colab benchmark run |
| Successfully processed videos | 28 | Videos that completed download, pose extraction, and feature generation |
| Failed or unavailable videos | 8 | Videos that could not be accessed at run time |
| Segment rows | 65 | Multi-segment metadata rows preserved in the run |
| Feature windows | 1,178 | Numeric windows generated from accessible videos |
| Positive windows | 349 | Windows labeled positive for the target behavior |
| Negative windows | 829 | Windows labeled negative for the target behavior |
| Positive prevalence | 29.6% | 349 / 1,178 |

Unavailable videos:

- action_18
- action_21
- action_22
- action_26
- action_27
- action_30
- action_35
- action_8

Observed failure causes were YouTube access and availability issues such as private videos, unavailable videos, terminated account status, truncated YouTube ID, or download/access failure.

## Pipeline summary

1. Validate SSBD+ metadata and subject/video grouping.
2. Resolve accessible videos from metadata-only references.
3. Download each accessible video temporarily for pose extraction only.
4. Extract numeric pose keypoints.
5. Build per-video feature windows.
6. Merge accessible per-video feature tables into a benchmark matrix.
7. Run group-disjoint baseline evaluation.
8. Run leave-one-subject-out evaluation.
9. Run permutation testing for the primary linear baseline.
10. Extract model-native feature importance from the fitted logistic regression model.
11. Write numeric CSV/JSON/TXT/SVG artifacts and a safe final zip archive.

Raw videos are temporary in the pipeline and are deleted after pose extraction. The final artifact zip contains only safe numeric and report artifacts.

## Artifact safety / privacy policy

| Category | Policy |
| --- | --- |
| Raw videos | Temporary only; not persisted in final artifacts |
| Frames / images | Not included in the final result archive |
| MediaPipe `.task` files | Not included in the final result archive |
| Trained model binaries | Not included in the final result archive |
| Result zip | Safe numeric/report artifacts only |
| Version control | Do not commit raw media, model binaries, or result archives |

The final archive is restricted to numeric CSV/JSON/TXT/SVG outputs. It intentionally excludes raw videos, frames, images, MediaPipe `.task` files, trained model binaries, cache directories, data directories, output directories, artifact directories, and git metadata.

## Evaluation protocol

| Protocol | Folds / repetitions | Grouping rule | Reported metrics |
| --- | ---: | --- | --- |
| GroupKFold | 5 folds | All windows from a video stay in the same fold | AUROC, AUPRC, Brier, ECE |
| LOSO | 28 folds | One accessible video left out per fold | AUROC, AUPRC |
| Permutation test | 1,000 permutations | Logistic regression AUROC under label permutations | Observed AUROC, p-value |

The reported fold summaries are means with standard deviations across folds unless otherwise noted. The permutation p-value is an empirical Monte Carlo estimate.

## GroupKFold results

| Model | AUROC | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.659 ± 0.024 | 0.440 ± 0.137 | 0.232 | 0.197 |
| Random Forest | 0.591 ± 0.044 | 0.368 ± 0.106 | 0.220 | 0.151 |

## LOSO results

| Model | AUROC | AUPRC |
| --- | ---: | ---: |
| Logistic Regression | 0.665 ± 0.179 | 0.596 ± 0.296 |
| Random Forest | 0.571 ± 0.165 | 0.528 ± 0.280 |

## Permutation test

| Model | Metric | Observed value | Permutations | p-value |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | AUROC | 0.659495 | 1,000 | 0.000999 |

## Feature importance

The logistic regression model’s most prominent features were:

| Rank | Feature | Conservative interpretation |
| --- | --- | --- |
| 1 | `inter_wrist_distance_std` | Variability in wrist separation |
| 2 | `inter_wrist_distance_mean` | Average wrist separation |
| 3 | `wrist_x_periodicity_strength` | Repetitive horizontal wrist motion |
| 4 | `head_y_dominant_frequency_hz` | Dominant vertical head motion frequency |
| 5 | `wrist_x_dominant_frequency_hz` | Dominant horizontal wrist motion frequency |
| 6 | `head_y_periodicity_strength` | Repetitive vertical head motion |

These are model-native, non-causal importance signals. They describe what the fitted baseline used in this run; they do not establish mechanism, causation, or clinical meaning.

## Interpretation

The strongest signal in this benchmark came from repetitive upper-limb and head-motion structure, especially wrist-distance variability and periodicity. That is consistent with a proxy motor-behavior recognition problem built from public research video.

The results should be read as exploratory numeric evidence for a narrow research task. They do not support autism detection, autism screening, clinical triage, medical-device claims, or deployment claims.

## Limitations

- Only 28 of 36 metadata videos were accessible in this run.
- YouTube availability and access conditions can change over time.
- The number of independent videos is modest, which makes fold estimates unstable.
- LOSO variance is high, so single-fold values should not be overinterpreted.
- Pose-estimation noise, occlusion, compression, and tracking errors can affect features.
- The benchmark is non-clinical and non-diagnostic by design.
- The evaluation does not establish real-world generalization, screening validity, or clinical validity.

## Reproducibility notes

- The Colab workflow expects a zip with the layout:

  ```text
  ssbd_colab_package/
    ssbd-stimming-recognition/
    metadata/
  ```

- The recommended workflow reuses completed outputs when `--resume` is provided.
- Raw videos are downloaded only temporarily, then deleted after extraction.
- The final result zip contains only safe numeric/report artifacts.
- The benchmark notebook documents the one-command Colab invocation.
- A packaging helper script is available to build the upload zip without including raw media or model artifacts.

## Conclusion

This completed benchmark run shows that the repository can produce reproducible, privacy-conscious numeric outputs for accessible SSBD+ videos. The run remains strictly research-only. It is not diagnostic, not screening, not clinical validation, and not deployment-ready.
