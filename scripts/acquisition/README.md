# Acquisition scripts

`resolve_videos.py` is an offline Phase 0 CLI scaffold for previewing or writing a CSV access report from a local manifest. It performs no network calls and contains no download implementation.

The default behavior is a dry run:

```bash
python scripts/acquisition/resolve_videos.py --manifest path/to/local_manifest.csv
```

The local manifest must contain `video_id`, `source`, and `url_or_manifest_reference` columns. `annotation_status` and `notes` are optional. Accepted source values are `SSBD+` and `original SSBD`.

Use `--output path/to/report.csv` to write the same report shown in the preview. Do not place output or future raw-media paths under git-tracked directories. Raw media belongs only in ignored temporary or `data/raw/` storage and must later follow the process-then-delete policy.

`--execute` is reserved for a later acquisition implementation and currently exits without downloading anything.
