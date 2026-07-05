# Stage G: Colab Distilled Multi-Scale Skeleton Temporal Feature Learning (D-MS-STF) results workflow

Stage G runs the Stage F Distilled Multi-Scale Skeleton Temporal Feature
Learning (D-MS-STF) evaluator on the numeric artifacts already produced by the
full SSBD+ benchmark. It does not download videos, decode frames, run
MediaPipe, or require a MediaPipe `.task` file.

Here, distillation means a fold-local teacher/student procedure: a teacher
model produces soft training targets inside the training fold, and a student
model is trained from hard labels plus these teacher-derived soft labels. No
test-fold labels or test-fold groups are used to generate distillation targets.

The documented final Stage G run was executed on the accessible-video SSBD+
benchmark cohort with 28 processed videos, 1,178 feature windows, 349 positive
windows, 829 negative windows, 48 total features after augmentation, 42
`ms_*` multi-scale features, and 1,000 within-group permutations.

Final empirical conclusion: mixed results. D-MS-STF was statistically above the
within-group permutation null, did not outperform the current logistic baseline
on AUROC, provided a small GroupKFold AUPRC improvement, and showed somewhat
better GroupKFold calibration than logistic. Teacher-only achieved better
calibration/Brier in some settings. Treat D-MS-STF as a tested research method
and ablation framework, not as a superior or SOTA model.

## Required inputs

Use the extracted privacy-safe benchmark artifacts:

- `features/all_features.csv`, or an equivalent numeric feature/window CSV
- `keypoints/`, the directory of per-video numeric keypoint CSVs, or one merged
  keypoint CSV

The feature table must contain `video_id`, `window_end_s`, `label`, and at least
one existing numeric baseline feature. Keypoint CSVs use the repository schema:
`video_id, frame_index, timestamp_s, landmark_index, x, y, z, confidence`.

If the feature table already has columns beginning with `ms_`, the workflow
copies it into the result directory and does not read keypoints. Otherwise it
infers the sample rate independently for each video from numeric timestamps and
adds aligned trailing 1 s, 2 s, and 4 s features at each `window_end_s`. Use
`--sample-rate-hz` only when the extraction rate is known and fixed.

## One Colab cell

Edit the four path variables. `ARTIFACT_ROOT` should point to the extracted
artifact directory from the existing benchmark run. Keep `SMOKE = True` for an
initial validation; set it to `False` for the 1,000-permutation final run.

```python
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/mohamad679/ssbd-stimming-recognition.git"
REPO_DIR = Path("/content/ssbd-stimming-recognition")
MOUNT_DRIVE = True
SMOKE = True

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("Google Drive mount is unavailable; using local Colab paths.")

if (REPO_DIR / ".git").is_dir():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "main"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"], check=True)
else:
    subprocess.run(["git", "clone", "--branch", "main", REPO_URL, str(REPO_DIR)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR)], check=True)

# Example layout from the full benchmark's extracted artifacts.
ARTIFACT_ROOT = Path("/content/drive/MyDrive/ssbd_benchmark_artifacts")
KEYPOINTS = ARTIFACT_ROOT / "keypoints"                  # CSV file or directory
FEATURE_CSV = ARTIFACT_ROOT / "features" / "all_features.csv"
RESULT_DIR = Path("/content/drive/MyDrive/stage_g_results")
ZIP_PATH = Path("/content/drive/MyDrive/ssbd_stage_g_final_results.zip")

command = [
    sys.executable,
    str(REPO_DIR / "scripts/benchmark/run_stage_g_d_ms_stf_colab.py"),
    "--feature-csv", str(FEATURE_CSV),
    "--keypoints", str(KEYPOINTS),
    "--output-dir", str(RESULT_DIR),
    "--group-splits", "5",
    "--inner-splits", "3",
    "--n-estimators", "200",
    "--zip-path", str(ZIP_PATH),
]
if SMOKE:
    command += ["--smoke", "--smoke-permutations", "2"]  # 5 is also supported
else:
    command += ["--n-permutations", "1000"]

subprocess.run(command, cwd=REPO_DIR, check=True)
```

The base package install intentionally omits benchmark extras: Stage G needs
NumPy and scikit-learn, but not `yt-dlp`, OpenCV, MediaPipe, or FFmpeg.

## Outputs

`RESULT_DIR` contains:

```text
stage_g_results/
  features_with_ms.csv
  aggregate_metrics.csv
  fold_metrics.csv
  report.json
```

With `--zip-path /path/to/ssbd_stage_g_final_results.zip`, the final Colab run
can also emit an explicitly named safe archive:

```text
ssbd_stage_g_final_results.zip
```

If `--zip-path` is omitted and `--create-zip` is used, the default name is a
sibling `<output-dir>.zip`. The zip builder includes only CSV, JSON, and TXT
files and excludes video, image, frame, `.task`, and model-binary patterns. It
also excludes safe-looking files inside directories named `raw`, `video(s)`,
`frame(s)`, `image(s)`, or `model(s)`. The zip is a generated result and must
not be committed.

The helper explicitly invokes `scripts/run_distilled_ms_stf.py` with both
`group_kfold` and `loso`. The final default is 1,000 within-group permutations;
`--smoke` resolves to 2 permutations by default and accepts 2 or 5 through
`--smoke-permutations`.

If `ms_*` columns already exist, `--keypoints` can be omitted:

```bash
python3 scripts/benchmark/run_stage_g_d_ms_stf_colab.py \
  --feature-csv /path/to/features_with_ms.csv \
  --output-dir /path/to/stage_g_results \
  --smoke --zip-path /path/to/ssbd_stage_g_final_results.zip
```

Generated artifacts from the final Colab run commonly include:

- `aggregate_metrics.csv`
- `fold_metrics.csv`
- `report.json`
- `features_with_ms.csv`
- `ssbd_stage_g_final_results.zip`

These are generated artifacts and should not be committed unless explicitly
intended.

## Final reported results

| Method | Distillation | Multi-scale | GroupKFold AUROC | GroupKFold AUPRC | LOSO AUROC | LOSO AUPRC | Permutation p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Logistic Baseline | No | No | 0.659 | 0.440 | 0.665 | 0.596 | — |
| Current Random Forest | No | No | 0.604 | 0.387 | 0.575 | 0.531 | — |
| MS-STF only | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| Teacher only | No | Yes | 0.637 | 0.449 | 0.607 | 0.606 | — |
| Student hard | No | Yes | 0.622 | 0.436 | 0.641 | 0.575 | — |
| D-MS-STF proposed | Yes | Yes | 0.625 | 0.447 | 0.633 | 0.590 | 0.000999 |

| Protocol | Model | Brier | ECE |
| --- | --- | ---: | ---: |
| GroupKFold | Logistic Regression | 0.232 | 0.197 |
| GroupKFold | Teacher only | 0.208 | 0.139 |
| GroupKFold | D-MS-STF proposed | 0.223 | 0.166 |
| LOSO | Logistic Regression | 0.238 | 0.286 |
| LOSO | Teacher only | 0.233 | 0.287 |
| LOSO | D-MS-STF proposed | 0.244 | 0.279 |

## Guardrails and limitations

- Inputs and outputs contain numeric pose/features and group identifiers; treat
  them as research data and keep Drive access appropriately restricted.
- The helper requires keypoint rows for every feature-table `video_id` when
  augmentation is needed and fails rather than silently dropping windows.
- Per-video rate inference requires at least two distinct timestamped frames.
- Existing `ms_*` columns are reused as supplied; their provenance is not
  reconstructed or compared with the requested scales.
- A 1,000-permutation run repeatedly fits nested models and can take a long time
  in Colab. Run smoke mode first and keep the runtime connected for the final
  run.
- This remains non-diagnostic motor-behavior recognition research, not clinical
  screening, clinical validation, or a deployment workflow.
