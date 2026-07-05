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
- Stage G feature set: 48 total features after augmentation
- Multi-scale feature count: 42 `ms_*` features
- Permutation test count for the final D-MS-STF run: 1,000

The strongest model signal in this run came from wrist-distance variability and repetitive wrist/head movement features. That is consistent with a non-diagnostic motor-behavior recognition task and should not be reframed as autism detection or screening.

Stage G completed a real D-MS-STF run on the accessible-video SSBD+ benchmark.
The final empirical picture is mixed: D-MS-STF was statistically above the
within-group permutation null, did not beat the current logistic baseline on
AUROC, slightly improved GroupKFold AUPRC, and showed somewhat better
GroupKFold calibration than logistic. Teacher-only achieved better
calibration/Brier in some settings. This supports D-MS-STF as a valid proposed
research method and ablation framework, not as a superior or SOTA model.

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
| Total features after Stage G augmentation | 48 | Final feature table used for the D-MS-STF comparison |
| `ms_*` multi-scale features | 42 | Added trailing 1 s, 2 s, and 4 s temporal summaries |

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
9. Augment the feature table with aligned multi-scale numeric temporal features.
10. Run Stage G ablations for MS-STF only, teacher-only, student-hard, and D-MS-STF.
11. Run 1,000 within-group permutations for the final D-MS-STF comparison.
12. Extract model-native feature importance from the fitted logistic regression model.
13. Write numeric CSV/JSON/TXT/SVG artifacts and a safe final zip archive.

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
| LOSO | 28 folds | One accessible video left out per fold | AUROC, AUPRC, Brier, ECE |
| Permutation test | 1,000 permutations | D-MS-STF GroupKFold AUROC under within-video label permutations | Observed AUROC, p-value |

The reported fold summaries are means with standard deviations across folds unless otherwise noted. The permutation p-value is an empirical Monte Carlo estimate.

## Final Stage G discrimination comparison

| Method | Distillation | Multi-scale | GroupKFold AUROC | GroupKFold AUPRC | LOSO AUROC | LOSO AUPRC | Permutation p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Logistic Baseline | No | No | 0.659 | 0.440 | 0.665 | 0.596 | — |
| Current Random Forest | No | No | 0.604 | 0.387 | 0.575 | 0.531 | — |
| MS-STF only | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| Teacher only | No | Yes | 0.637 | 0.449 | 0.607 | 0.606 | — |
| Student hard | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| D-MS-STF proposed | Yes | Yes | 0.625 | 0.447 | 0.633 | 0.590 | 0.000999 |

## Calibration snapshot

| Protocol | Model | Brier | ECE |
| --- | --- | ---: | ---: |
| GroupKFold | Logistic Regression | 0.232 | 0.197 |
| GroupKFold | Teacher only | 0.208 | 0.139 |
| GroupKFold | D-MS-STF proposed | 0.223 | 0.166 |
| LOSO | Logistic Regression | 0.238 | 0.286 |
| LOSO | Teacher only | 0.233 | 0.287 |
| LOSO | D-MS-STF proposed | 0.244 | 0.279 |

## Permutation test

| Model | Metric | Observed value | Permutations | p-value |
| --- | --- | ---: | ---: | ---: |
| D-MS-STF proposed | GroupKFold AUROC | 0.625 | 1,000 | 0.000999 |

## Final Stage G interpretation

- Stage G completed a real D-MS-STF run on the accessible-video SSBD+ benchmark.
- D-MS-STF was statistically above the within-group permutation null.
- D-MS-STF did not outperform the current logistic baseline on AUROC.
- D-MS-STF provided a small GroupKFold AUPRC improvement over the logistic baseline.
- D-MS-STF showed somewhat better GroupKFold calibration than logistic, while teacher-only achieved the best Brier/ECE values in some settings.
- Overall empirical conclusion: mixed results. D-MS-STF is a valid proposed method and ablation framework, but not a demonstrated superior model on this small accessible-video cohort.

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

The strongest signal in this benchmark came from repetitive upper-limb and
head-motion structure, especially wrist-distance variability and periodicity.
That is consistent with a proxy motor-behavior recognition problem built from
public research video.

The results should be read as exploratory numeric evidence for a narrow
research task. They do not support autism detection, autism screening,
clinical triage, medical-device claims, or deployment claims.

## Limitations

- Only 28 of 36 metadata videos were accessible in this run.
- YouTube availability and access conditions can change over time.
- The number of independent videos is modest, which makes fold estimates unstable.
- LOSO variance is high, so single-fold values should not be overinterpreted.
- The D-MS-STF comparison is mixed and does not establish overall superiority to the logistic baseline.
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
- The final Colab run produced generated artifacts such as `aggregate_metrics.csv`, `fold_metrics.csv`, `report.json`, `features_with_ms.csv`, and `ssbd_stage_g_final_results.zip`; these are generated outputs and should not be committed unless explicitly intended.

## Conclusion

This completed benchmark run shows that the repository can produce
reproducible, privacy-conscious numeric outputs for accessible SSBD+ videos,
including a real Stage G D-MS-STF ablation on the accessible-video cohort. The
run remains strictly research-only. It is not diagnostic, not screening, not
clinical validation, and not deployment-ready. The final empirical conclusion
is mixed and does not justify superiority or SOTA claims for D-MS-STF.
