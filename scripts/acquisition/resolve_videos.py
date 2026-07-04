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
    AccessReportRow,
    DatasetSource,
    ManifestEntry,
    load_ssbdplus_csv,
    render_access_report_csv,
    write_access_report_csv,
)


SSBDPLUS_COLUMNS = {
    "xml_file_name",
    "youtube_video_url",
    "action_start_time",
    "action_end_time",
    "action_category",
}
SSBDPLUS_DRY_RUN_NOTE = "Metadata-only dry run; video availability not verified."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a metadata-only SSBD access report without downloading."
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
        help="local generic or observed SSBD+ segment CSV; omit for an empty report",
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


def read_manifest_report_rows(path: Path) -> list[AccessReportRow]:
    """Read either supported manifest schema and return unattempted report rows."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        columns = set(csv.DictReader(handle).fieldnames or ())

    if SSBDPLUS_COLUMNS.issubset(columns):
        segments = load_ssbdplus_csv(path)
        grouped: dict[tuple[str, str | None, str], int] = {}
        for segment in segments:
            key = (segment.video_id, segment.annotation_file, segment.url)
            grouped[key] = grouped.get(key, 0) + 1

        return [
            AccessReportRow(
                attempted_video_id=video_id,
                source=DatasetSource.SSBDPLUS,
                url_or_manifest_reference=url,
                download_status="not_attempted",
                annotation_status="present",
                usable_segment_count=segment_count,
                notes=SSBDPLUS_DRY_RUN_NOTE,
            )
            for (video_id, _annotation_file, url), segment_count in grouped.items()
        ]

    return [entry.to_access_report_row() for entry in read_local_manifest(path)]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        print(
            "Download execution is intentionally not implemented in Phase 0 scaffolding.",
            file=sys.stderr,
        )
        return 2

    rows = read_manifest_report_rows(args.manifest) if args.manifest else []

    if args.output:
        write_access_report_csv(rows, args.output)
        print(f"Dry run: wrote access report to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(render_access_report_csv(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
