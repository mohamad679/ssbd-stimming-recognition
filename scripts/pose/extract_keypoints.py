#!/usr/bin/env python3
"""Extract numeric pose keypoints from one explicit local temporary video."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.pose import ExtractionConfig, extract_pose_keypoints  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or extract one local temporary video. This command never "
            "downloads data, crawls manifests, or batch-processes SSBD."
        )
    )
    parser.add_argument("video_id")
    parser.add_argument("input_video_path", type=Path)
    parser.add_argument("output_csv_path", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform extraction (default is a validation-only dry run)",
    )
    parser.add_argument("--keep-input", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExtractionConfig(
        video_id=args.video_id,
        input_video_path=args.input_video_path,
        output_csv_path=args.output_csv_path,
        delete_input_after_success=not args.keep_input,
        max_frames=args.max_frames,
    )
    if not args.execute:
        print(
            "Dry run only: local paths validated; no video was decoded and no file "
            "was written. Use --execute for this single local temporary video.",
            file=sys.stderr,
        )
        return 0

    print(
        "WARNING: local temporary video only; numeric CSV is the sole persisted output.",
        file=sys.stderr,
    )
    rows = extract_pose_keypoints(config)
    print(f"Wrote {len(rows)} numeric keypoint rows to {args.output_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
