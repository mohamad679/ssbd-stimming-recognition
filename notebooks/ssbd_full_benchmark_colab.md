# SSBD+ full benchmark in one Colab command

This workflow is for privacy-preserving, non-diagnostic motor-behavior research. It
does not provide clinical or screening conclusions. Downloaded videos are temporary,
are deleted after pose extraction, and are never included in the result archive.

## Prepare and upload

Prepare one zip named `ssbd_colab_package.zip` with this layout:

```text
ssbd_colab_package/
  ssbd-stimming-recognition/   # this repository
  metadata/                    # the full SSBD+ metadata CSV
```

Build the zip from the repository root, then upload it to the Colab session root
(`/content`). Do not add raw videos, frames, images, MediaPipe `.task` files, model
binaries, caches, outputs, or the benchmark result zip to the package.

Recommended packaging command:

```bash
python3 scripts/packaging/build_ssbd_colab_package.py --repo-root . --metadata-dir /path/to/ssbdplus-metadata --output-zip ssbd_colab_package.zip
```

Upload that zip to the Colab session root (`/content`).

## Run one command

Run this single Colab shell command. Re-running the same command resumes completed
per-video keypoint and feature outputs.

```bash
!unzip -q -o /content/ssbd_colab_package.zip -d /content && pip install -q -e "/content/ssbd_colab_package/ssbd-stimming-recognition[benchmark]" && python /content/ssbd_colab_package/ssbd-stimming-recognition/scripts/benchmark/run_full_ssbdplus_colab_benchmark.py --repo-root /content/ssbd_colab_package/ssbd-stimming-recognition --metadata-dir /content/ssbd_colab_package/metadata --work-dir /content/ssbdplus-work --output-zip /content/ssbdplus-results.zip --n-splits 5 --n-permutations 1000 --max-height 720 --resume --execute
```

The final `/content/ssbdplus-results.zip` contains numeric CSVs, aggregate reports,
JSON manifests, and abstract skeleton SVGs only. It excludes videos, frames, images,
MediaPipe `.task` files, and trained model files. `failure_report.csv` records any
video or evaluation stage that could not be completed, while `run_summary.json`
summarizes the run.
