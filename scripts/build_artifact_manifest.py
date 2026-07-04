#!/usr/bin/env python3
"""Build a provenance manifest for explicit numeric artifact files only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.evaluation import (  # noqa: E402
    build_artifact_records,
    write_artifact_manifest,
)


SUSPICIOUS_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".jpg",
    ".jpeg",
    ".png",
    ".pkl",
    ".joblib",
    ".pt",
    ".pth",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a provenance manifest for explicit numeric artifact files only. "
            "This command does not crawl datasets, read videos, run MediaPipe, or "
            "record model artifacts."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--artifact-type", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--notes",
        default=None,
        help="optional note to attach to every artifact record",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the manifest (default is a validation-only dry run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _reject_suspicious_paths(args.paths)
        records = build_artifact_records(
            args.paths,
            artifact_type=args.artifact_type,
            notes=args.notes,
        )
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        print(
            f"Dry run: would write {len(records)} {args.artifact_type} artifact record(s) "
            f"to {args.output}. No file was written. Use --execute to write the manifest."
        )
        for record in records:
            print(
                f"- path={record.path} size_bytes={record.size_bytes} "
                f"sha256={record.sha256} artifact_type={record.artifact_type}"
            )
        return 0

    write_artifact_manifest(args.output, records)
    print(f"Wrote artifact manifest with {len(records)} record(s) to {args.output}")
    return 0


def _reject_suspicious_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.suffix.lower() in SUSPICIOUS_EXTENSIONS:
            raise ValueError(
                "refusing suspicious raw-media or model-artifact path: "
                f"{path} (extension {path.suffix.lower()} is not allowed)"
            )


if __name__ == "__main__":
    raise SystemExit(main())
