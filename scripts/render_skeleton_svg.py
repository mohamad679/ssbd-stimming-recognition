#!/usr/bin/env python3
"""Render abstract SVG skeletons from numeric keypoint CSV files only."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.evaluation import validate_svg_file  # noqa: E402
from ssbd_behavior.interpretability import (  # noqa: E402
    render_sequence_summary_svg,
    render_skeleton_svg,
)
from ssbd_behavior.pose import read_keypoints_csv  # noqa: E402


SUSPICIOUS_MEDIA_EXTENSIONS = {
    ".avi",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".png",
    ".webp",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a privacy-safe abstract skeleton SVG from numeric keypoint CSV "
            "data only. This command does not read videos or images and does not run "
            "MediaPipe."
        )
    )
    parser.add_argument("input_csv", type=Path, help="numeric keypoint CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="explicit SVG output path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the SVG output; omitted by default for dry-run mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = _validate_input_csv_path(args.input_csv)
        output_path = _validate_output_path(args.output)
        rows = read_keypoints_csv(input_path)
        grouped_rows = _group_rows_by_frame(rows)
        svg_text = (
            render_skeleton_svg(grouped_rows[0])
            if len(grouped_rows) == 1
            else render_sequence_summary_svg(grouped_rows)
        )
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        print(
            f"Dry run: rendered SVG text for {len(grouped_rows)} pose(s); "
            f"use --execute to write {output_path}"
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_text, encoding="utf-8")

    try:
        validate_svg_file(output_path)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote and validated SVG: {output_path}")
    return 0


def _validate_input_csv_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in SUSPICIOUS_MEDIA_EXTENSIONS:
        raise ValueError(f"media/image inputs are not supported: {path}")
    if suffix != ".csv":
        raise ValueError(f"input path must point to a numeric keypoint .csv file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"keypoint CSV does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"keypoint CSV path must be a file: {path}")
    return path


def _validate_output_path(path: Path) -> Path:
    if path.suffix.lower() != ".svg":
        raise ValueError(f"output path must have a .svg extension: {path}")
    return path


def _group_rows_by_frame(rows):
    if not rows:
        raise ValueError("keypoint CSV must contain at least one keypoint row")

    video_ids = {row.video_id for row in rows}
    if len(video_ids) != 1:
        raise ValueError("keypoint CSV must contain exactly one video_id")

    frames: OrderedDict[int, list] = OrderedDict()
    for row in rows:
        frames.setdefault(row.frame_index, []).append(row)
    return list(frames.values())


if __name__ == "__main__":
    raise SystemExit(main())
