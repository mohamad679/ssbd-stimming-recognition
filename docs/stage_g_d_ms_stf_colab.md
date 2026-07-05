# Stage G: Colab D-MS-STF results workflow

Stage G runs the Stage F D-MS-STF evaluator on the numeric artifacts already
produced by the full SSBD+ benchmark. It does not download videos, decode
frames, run MediaPipe, or require a MediaPipe `.task` file.

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

command = [
    sys.executable,
    str(REPO_DIR / "scripts/benchmark/run_stage_g_d_ms_stf_colab.py"),
    "--feature-csv", str(FEATURE_CSV),
    "--keypoints", str(KEYPOINTS),
    "--output-dir", str(RESULT_DIR),
    "--group-splits", "5",
    "--inner-splits", "3",
    "--n-estimators", "200",
    "--create-zip",
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

With `--create-zip`, a sibling `stage_g_results.zip` is also written. The zip
builder includes only CSV, JSON, and TXT files and excludes video, image,
frame, `.task`, and model-binary patterns. It also excludes safe-looking files
inside directories named `raw`, `video(s)`, `frame(s)`, `image(s)`, or
`model(s)`. The zip is a generated result and must not be committed.

The helper explicitly invokes `scripts/run_distilled_ms_stf.py` with both
`group_kfold` and `loso`. The final default is 1,000 within-group permutations;
`--smoke` resolves to 2 permutations by default and accepts 2 or 5 through
`--smoke-permutations`.

If `ms_*` columns already exist, `--keypoints` can be omitted:

```bash
python3 scripts/benchmark/run_stage_g_d_ms_stf_colab.py \
  --feature-csv /path/to/features_with_ms.csv \
  --output-dir /path/to/stage_g_results \
  --smoke --create-zip
```

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
