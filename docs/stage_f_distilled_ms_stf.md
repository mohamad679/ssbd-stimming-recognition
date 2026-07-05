# Stage F: D-MS-STF

**D-MS-STF** means **Distilled Multi-Scale Skeleton Temporal Feature Learning**.
It is a proposed, non-diagnostic motor-behavior recognition method operating on
numeric pose keypoints and numeric window features.

## Method

The processing path is:

```text
numeric pose keypoints/features
  -> aligned 1 s, 2 s, and 4 s temporal features
  -> outer-train-only teacher
  -> inner group-cross-fitted teacher probabilities
  -> lightweight logistic student
  -> untouched outer-test videos
```

The multi-scale feature implementation is
`ssbd_behavior.features.multiscale_temporal_feature_dict`. Each scale is a
trailing interval ending at the same `reference_end_s`. It emits:

- inter-wrist distance mean, standard deviation, and range
- wrist velocity and acceleration summaries
- head vertical motion
- dominant frequency and periodicity strength
- left/right motion symmetry
- pose confidence, observed-frame coverage, and missingness
- motion energy

Missing landmarks are excluded from pairwise motion calculations and are
represented explicitly by quality features. Empty scales return finite zeros
for motion summaries and `1.0` missingness. No raw frame is required.

The default teacher is an `ExtraTreesClassifier`; random forest and histogram
gradient boosting are configurable alternatives. The student is scaled
logistic regression. Optional sigmoid calibration uses an additional
group-disjoint split inside the data currently available to the teacher.

## Distillation objective

For hard label `y`, inner-cross-fitted teacher probability `p`, hard-label
weight `alpha`, and temperature `T`, the student target is:

```text
q = alpha * y + (1 - alpha) * sigmoid(logit(p) / T)
```

Scikit-learn classifiers do not directly accept fractional targets. The
implementation duplicates each training row with targets zero and one, then
uses weights `1-q` and `q`. This is algebraically the Bernoulli log-loss for
soft target `q`. Per-sample class-balance weights are computed from the outer
training hard labels and applied equally to each duplicate.

## Leakage safety

Distillation is nested inside every outer fold. For each GroupKFold or LOSO
fold, the implementation:

1. Selects fixed feature columns by name and passes only outer-training rows to
   learned preprocessing and model fitting.
2. Fits every student scaler only on outer-training rows.
3. Partitions the outer-training videos with inner `GroupKFold`.
4. Trains each inner teacher without its inner-validation videos.
5. Generates one teacher probability for every outer-training row only from a
   teacher that did not train on that row's video.
6. Fits the student on outer-training rows and cross-fitted probabilities.
7. Uses the student directly on untouched outer-test videos.

The JSON report records outer and inner group membership for audit. Runtime
checks reject group overlap. The following workflow is prohibited:

```text
full data -> one teacher -> global soft labels -> cross-validation
```

No scaler, selector, teacher, student, or calibrator may fit on outer-test data.
Temperature is a fixed configuration value, not a parameter selected using the
outer test fold.

## Ablations and outputs

The runner evaluates:

- `current_logistic_baseline`: non-multi-scale feature columns only
- `current_random_forest`: existing random-forest baseline on those same columns
- `ms_stf_only`: logistic regression with all features
- `teacher_only`: fold-local teacher with all features
- `student_hard`: lightweight student with hard labels only
- `d_ms_stf`: lightweight student with hard and cross-fitted soft labels
- optional `teacher_calibrated` and `d_ms_stf_calibrated` variants

Both GroupKFold and LOSO report AUROC, AUPRC, Brier score, and fixed-bin ECE.
Single-class test folds retain Brier/ECE and mark AUROC/AUPRC unavailable. The
optional permutation test shuffles labels within videos and reruns the complete
fold-local D-MS-STF training path.

The output directory contains only:

- `fold_metrics.csv`
- `aggregate_metrics.csv`
- `report.json`

These are numeric/aggregate artifacts plus group identifiers already present
in the input feature table. Trained estimators are never serialized.

## Colab workflow

Use existing extracted keypoint and feature CSVs. Do not download raw videos for
Stage F and do not upload videos, frames, images, or MediaPipe model files to the
repository.

```bash
git clone https://github.com/mohamad679/ssbd-stimming-recognition.git
cd ssbd-stimming-recognition
python3 -m pip install -e .
```

If the existing feature table does not yet contain `ms_` columns, add them from
an existing numeric keypoint CSV. The example assumes the feature table has
`video_id` and `window_end_s` columns and the keypoint CSV contains all videos.

```python
import csv
from collections import defaultdict
from pathlib import Path

from ssbd_behavior.features import multiscale_temporal_feature_dict
from ssbd_behavior.pose import read_keypoints_csv

keypoints_by_video = defaultdict(list)
for point in read_keypoints_csv(Path("keypoints.csv")):
    keypoints_by_video[point.video_id].append(point)

with Path("features.csv").open(newline="", encoding="utf-8-sig") as source:
    rows = list(csv.DictReader(source))
    original_columns = tuple(rows[0])

for row in rows:
    row.update(
        multiscale_temporal_feature_dict(
            keypoints_by_video[row["video_id"]],
            sample_rate_hz=30.0,
            scales_s=(1.0, 2.0, 4.0),
            reference_end_s=float(row["window_end_s"]),
        )
    )

multi_scale_columns = tuple(name for name in rows[0] if name.startswith("ms_"))
with Path("features_stage_f.csv").open("w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=original_columns + multi_scale_columns)
    writer.writeheader()
    writer.writerows(rows)
```

Use the actual extraction sample rate rather than assuming 30 Hz. Then run:

```bash
python3 scripts/run_distilled_ms_stf.py features_stage_f.csv \
  --output-dir stage_f_results \
  --group-splits 5 \
  --inner-splits 3 \
  --teacher extra_trees \
  --alpha 0.5 \
  --temperature 1.0 \
  --n-permutations 1000
```

Run without `--include-calibrated-teacher` first. Calibration needs enough
videos and both labels in every nested calibration partition.

## Comparison target

| Method | Distillation | Multi-scale | GroupKFold AUROC | GroupKFold AUPRC | LOSO AUROC | LOSO AUPRC | Permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Logistic Baseline | No | No | 0.659 | 0.440 | 0.665 | 0.596 | 0.000999 |
| Current Random Forest | No | No | 0.591 | 0.368 | 0.571 | 0.528 | unavailable |
| MS-STF only | No | Yes | pending | pending | pending | pending | pending |
| D-MS-STF proposed | Yes | Yes | pending | pending | pending | pending | pending |

Existing baseline values are historical benchmark references. Stage F values
must come from a fresh, versioned run on the same feature-table cohort and
group definitions before comparison.

## Limitations and guardrails

- This is non-diagnostic motor-behavior recognition research. It is not autism
  diagnosis, clinical screening, or clinical validation.
- It is not a medical device, surveillance system, deployment claim, or
  production system.
- No SOTA claim is supported by this scaffold or by an unfilled comparison row.
- Coordinate-derived velocities are sensitive to pose scale, camera motion,
  extraction frame rate, and missing observations.
- FFT features assume approximately regular sampling after missing observations
  are excluded; severe dropout can make them unreliable.
- Cross-fitted probabilities reduce leakage but do not correct dataset bias,
  label ambiguity, repeated-person dependence, or unavailable videos.
- Calibration can be unstable in small nested folds and is optional.
- Permutation testing is computationally expensive because it refits every
  inner and outer model.
- Raw videos must remain temporary and must not be committed. Final outputs
  must exclude videos, frames, images, MediaPipe `.task` files, and trained
  model binaries. Privacy-safe numeric CSV/JSON reports and abstract SVGs are
  the permitted artifact types.
