#!/usr/bin/env python3
"""Preview SSBD manifest entries as an access report without network access."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.acquisition import (  # noqa: E402
    DatasetSource,
    ManifestEntry,
    render_access_report_csv,
    write_access_report_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview a local SSBD manifest as a CSV access report."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="preview or write unattempted report rows (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_false",
        dest="dry_run",
        help="reserved for future explicit download execution",
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="local CSV manifest; omit to preview an empty report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional destination for the CSV access report",
    )
    return parser


def read_local_manifest(path: Path) -> list[ManifestEntry]:
    """Read metadata from a local CSV file without resolving any references."""

    entries: list[ManifestEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "source", "url_or_manifest_reference"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            entries.append(
                ManifestEntry(
                    video_id=row["video_id"],
                    source=DatasetSource(row["source"]),
                    reference=row["url_or_manifest_reference"],
                    annotation_status=row.get("annotation_status") or "unknown",
                    notes=row.get("notes") or "",
                )
            )
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        print(
            "Download execution is intentionally not implemented in Phase 0 scaffolding.",
            file=sys.stderr,
        )
        return 2

    entries = read_local_manifest(args.manifest) if args.manifest else []
    rows = [entry.to_access_report_row() for entry in entries]
    report = render_access_report_csv(rows)

    if args.output:
        write_access_report_csv(rows, args.output)
        print(f"Dry run: wrote access report to {args.output}", file=sys.stderr)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
