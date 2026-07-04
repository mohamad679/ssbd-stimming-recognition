#!/usr/bin/env python3
"""Build a numeric feature table from an explicit local keypoint CSV."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.acquisition import load_ssbdplus_csv  # noqa: E402
from ssbd_behavior.features import (  # noqa: E402
    WindowSpec,
    build_feature_rows_for_video,
    write_feature_table_csv,
)
from ssbd_behavior.pose import read_keypoints_csv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build numeric window features from one explicit local keypoint CSV. "
            "This command never reads videos, runs MediaPipe, or accesses a network."
        )
    )
    parser.add_argument("input_keypoint_csv", type=Path)
    parser.add_argument("output_feature_csv", type=Path)
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--window-size-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.5)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the feature CSV (default is a validation-only dry run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keypoints = read_keypoints_csv(args.input_keypoint_csv)
    annotations = (
        load_ssbdplus_csv(args.annotations_csv) if args.annotations_csv else []
    )
    spec = WindowSpec(args.window_size_s, args.stride_s)

    rows_by_video = defaultdict(list)
    for row in keypoints:
        rows_by_video[row.video_id].append(row)

    feature_rows = []
    for video_id in sorted(rows_by_video):
        feature_rows.extend(
            build_feature_rows_for_video(
                video_id=video_id,
                keypoint_rows=rows_by_video[video_id],
                annotations=annotations,
                spec=spec,
                sample_rate_hz=args.sample_rate_hz,
                minimum_overlap_fraction=args.minimum_overlap_fraction,
            )
        )

    if not args.execute:
        print(
            f"Dry run: validated {len(keypoints)} keypoints and would write "
            f"{len(feature_rows)} feature rows for {len(rows_by_video)} videos. "
            "No file was written. Use --execute to write the output.",
            file=sys.stderr,
        )
        return 0

    write_feature_table_csv(args.output_feature_csv, feature_rows)
    print(f"Wrote {len(feature_rows)} numeric feature rows to {args.output_feature_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
