#!/usr/bin/env python3
"""Run Stage G D-MS-STF from existing privacy-safe numeric SSBD+ artifacts."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.features import (  # noqa: E402
    DEFAULT_TEMPORAL_SCALES_S,
    multiscale_temporal_feature_dict,
)
from ssbd_behavior.pose import PoseKeypoint, read_keypoints_csv  # noqa: E402


SAFE_RESULT_SUFFIXES = frozenset({".csv", ".json", ".txt"})
FORBIDDEN_RESULT_SUFFIXES = frozenset(
    {
        ".avi",
        ".bin",
        ".bmp",
        ".ckpt",
        ".gif",
        ".h5",
        ".jpeg",
        ".jpg",
        ".joblib",
        ".keras",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".onnx",
        ".pb",
        ".pickle",
        ".pkl",
        ".png",
        ".pt",
        ".pth",
        ".safetensors",
        ".task",
        ".tflite",
        ".webm",
        ".webp",
    }
)
FORBIDDEN_RESULT_DIR_NAMES = frozenset(
    {"frame", "frames", "image", "images", "model", "models", "raw", "video", "videos"}
)
REQUIRED_RESULT_FILES = frozenset(
    {"aggregate_metrics.csv", "fold_metrics.csv", "report.json"}
)


@dataclass(frozen=True)
class WorkflowResult:
    feature_csv: Path
    result_files: tuple[Path, ...]
    zip_path: Path | None
    features_augmented: bool
    n_permutations: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Augment an existing numeric SSBD+ feature table with multi-scale "
            "keypoint features when needed, then run leakage-safe D-MS-STF. "
            "This command never downloads or reads videos and never runs MediaPipe."
        )
    )
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument(
        "--keypoints",
        type=Path,
        help=(
            "numeric keypoint CSV or directory of per-video CSVs; optional when "
            "the feature table already contains ms_* columns"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id-column", default="video_id")
    parser.add_argument("--window-end-column", default="window_end_s")
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--multiscale-prefix", default="ms_")
    parser.add_argument(
        "--scales-s",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPORAL_SCALES_S,
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        help="fixed extraction rate; by default it is inferred per video",
    )
    parser.add_argument("--group-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--n-permutations",
        type=int,
        help="defaults to 1000 for a final run or --smoke-permutations in smoke mode",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a small permutation count for an end-to-end validation run",
    )
    parser.add_argument(
        "--smoke-permutations", type=int, choices=(2, 5), default=2
    )
    parser.add_argument("--create-zip", action="store_true")
    parser.add_argument(
        "--zip-path",
        type=Path,
        help="safe result zip path; implies --create-zip and defaults beside output-dir",
    )
    return parser


def augment_multiscale_feature_csv(
    feature_csv: Path | str,
    output_csv: Path | str,
    *,
    keypoints: Path | str | None,
    video_id_column: str = "video_id",
    window_end_column: str = "window_end_s",
    multiscale_prefix: str = "ms_",
    scales_s: Iterable[float] = DEFAULT_TEMPORAL_SCALES_S,
    sample_rate_hz: float | None = None,
) -> bool:
    """Write a Stage G feature CSV and return whether multi-scale columns were added."""

    source = Path(feature_csv).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    columns, rows = _read_csv_rows(source)
    existing = tuple(name for name in columns if name.startswith(multiscale_prefix))
    if existing:
        _write_csv_rows(destination, columns, rows)
        return False

    for required in (video_id_column, window_end_column):
        if required not in columns:
            raise ValueError(f"feature table is missing required column {required!r}")
    if keypoints is None:
        raise ValueError(
            "--keypoints is required because the feature table has no "
            f"{multiscale_prefix}* columns"
        )

    scales = _validated_scales(scales_s)
    fixed_rate = (
        _validated_sample_rate(sample_rate_hz)
        if sample_rate_hz is not None
        else None
    )
    points_by_video = _load_keypoints_by_video(Path(keypoints))
    indexed_points: dict[str, tuple[list[PoseKeypoint], list[float], float]] = {}
    for video_id, video_points in points_by_video.items():
        ordered = sorted(
            video_points,
            key=lambda point: (point.timestamp_s, point.frame_index, point.landmark_index),
        )
        rate = fixed_rate or infer_sample_rate_hz(ordered)
        indexed_points[video_id] = (
            ordered,
            [point.timestamp_s for point in ordered],
            rate,
        )

    augmented_rows: list[dict[str, str]] = []
    multiscale_columns: tuple[str, ...] | None = None
    missing_videos: set[str] = set()
    max_scale = max(scales)
    for row_number, row in enumerate(rows, start=2):
        video_id = row[video_id_column].strip()
        if not video_id:
            raise ValueError(f"feature CSV row {row_number} has an empty video id")
        if video_id not in indexed_points:
            missing_videos.add(video_id)
            continue
        try:
            end_s = float(row[window_end_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature CSV row {row_number} has an invalid window end"
            ) from exc
        if not math.isfinite(end_s) or end_s < 0:
            raise ValueError(
                f"feature CSV row {row_number} has a non-finite or negative window end"
            )

        video_points, timestamps, rate = indexed_points[video_id]
        start_index = bisect_left(timestamps, end_s - max_scale)
        end_index = bisect_left(timestamps, end_s)
        features = multiscale_temporal_feature_dict(
            video_points[start_index:end_index],
            sample_rate_hz=rate,
            scales_s=scales,
            reference_end_s=end_s,
        )
        current_columns = tuple(features)
        if multiscale_columns is None:
            multiscale_columns = current_columns
        elif current_columns != multiscale_columns:
            raise RuntimeError("multi-scale feature schema changed between rows")
        augmented = dict(row)
        augmented.update({name: _format_number(value) for name, value in features.items()})
        augmented_rows.append(augmented)

    if missing_videos:
        missing = ", ".join(sorted(missing_videos))
        raise ValueError(f"keypoint input has no rows for feature-table videos: {missing}")
    if not augmented_rows or not multiscale_columns:
        raise ValueError("feature table must contain at least one data row")
    _write_csv_rows(destination, columns + multiscale_columns, augmented_rows)
    return True


def infer_sample_rate_hz(points: Sequence[PoseKeypoint]) -> float:
    """Infer a per-video sample rate from unique frame timestamps."""

    timestamps_by_frame: dict[int, float] = {}
    for point in points:
        timestamps_by_frame.setdefault(point.frame_index, point.timestamp_s)
    timestamps = [timestamps_by_frame[index] for index in sorted(timestamps_by_frame)]
    intervals = sorted(
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    )
    if not intervals:
        raise ValueError("at least two distinct timestamped frames are required per video")
    middle = len(intervals) // 2
    interval = (
        intervals[middle]
        if len(intervals) % 2
        else (intervals[middle - 1] + intervals[middle]) / 2.0
    )
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("could not infer a positive sample interval")
    return 1.0 / interval


def create_safe_results_zip(
    result_dir: Path | str, output_zip: Path | str
) -> tuple[str, ...]:
    """Package allowlisted numeric reports while excluding media and model files."""

    root = Path(result_dir).expanduser().resolve()
    destination = Path(output_zip).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"result directory does not exist: {root}")
    if destination == root or root in destination.parents:
        raise ValueError("output zip must be outside the result directory")

    members: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        parts = {part.lower() for part in relative.parts[:-1]}
        suffix = path.suffix.lower()
        if parts & FORBIDDEN_RESULT_DIR_NAMES:
            continue
        if suffix in FORBIDDEN_RESULT_SUFFIXES or suffix not in SAFE_RESULT_SUFFIXES:
            continue
        members.append((path, relative.as_posix()))
    if not members:
        raise ValueError("result directory contains no safe result files")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, member in members:
                archive.write(path, member)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return tuple(member for _, member in members)


def run_stage_g_workflow(
    *,
    feature_csv: Path,
    keypoints: Path | None,
    output_dir: Path,
    video_id_column: str = "video_id",
    window_end_column: str = "window_end_s",
    group_column: str = "video_id",
    label_column: str = "label",
    multiscale_prefix: str = "ms_",
    scales_s: Iterable[float] = DEFAULT_TEMPORAL_SCALES_S,
    sample_rate_hz: float | None = None,
    group_splits: int = 5,
    inner_splits: int = 3,
    n_estimators: int = 200,
    n_permutations: int = 1000,
    zip_path: Path | None = None,
) -> WorkflowResult:
    """Augment numeric features, invoke the Stage F runner, and optionally zip results."""

    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stage_g_features = destination / "features_with_ms.csv"
    augmented = augment_multiscale_feature_csv(
        feature_csv,
        stage_g_features,
        keypoints=keypoints,
        video_id_column=video_id_column,
        window_end_column=window_end_column,
        multiscale_prefix=multiscale_prefix,
        scales_s=scales_s,
        sample_rate_hz=sample_rate_hz,
    )

    runner = REPOSITORY_ROOT / "scripts" / "run_distilled_ms_stf.py"
    command = [
        sys.executable,
        str(runner),
        str(stage_g_features),
        "--output-dir",
        str(destination),
        "--group-column",
        group_column,
        "--label-column",
        label_column,
        "--multiscale-prefix",
        multiscale_prefix,
        "--group-splits",
        str(group_splits),
        "--inner-splits",
        str(inner_splits),
        "--n-estimators",
        str(n_estimators),
        "--n-permutations",
        str(n_permutations),
        "--protocol",
        "group_kfold",
        "--protocol",
        "loso",
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    result_files = tuple(destination / name for name in sorted(REQUIRED_RESULT_FILES))
    missing = [path.name for path in result_files if not path.is_file()]
    if missing:
        raise RuntimeError("Stage F runner did not create: " + ", ".join(missing))
    archive = None
    if zip_path is not None:
        archive = zip_path.expanduser().resolve()
        create_safe_results_zip(destination, archive)
    return WorkflowResult(
        feature_csv=stage_g_features,
        result_files=result_files,
        zip_path=archive,
        features_augmented=augmented,
        n_permutations=n_permutations,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_permutations = args.n_permutations
    if n_permutations is None:
        n_permutations = args.smoke_permutations if args.smoke else 1000
    output_dir = args.output_dir.expanduser().resolve()
    create_zip = args.create_zip or args.zip_path is not None
    zip_path = args.zip_path
    if create_zip and zip_path is None:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"

    result = run_stage_g_workflow(
        feature_csv=args.feature_csv,
        keypoints=args.keypoints,
        output_dir=output_dir,
        video_id_column=args.video_id_column,
        window_end_column=args.window_end_column,
        group_column=args.group_column,
        label_column=args.label_column,
        multiscale_prefix=args.multiscale_prefix,
        scales_s=args.scales_s,
        sample_rate_hz=args.sample_rate_hz,
        group_splits=args.group_splits,
        inner_splits=args.inner_splits,
        n_estimators=args.n_estimators,
        n_permutations=n_permutations,
        zip_path=zip_path,
    )
    action = "added" if result.features_augmented else "reused existing"
    print(f"Stage G {action} {args.multiscale_prefix}* features: {result.feature_csv}")
    print(f"Permutation count: {result.n_permutations}")
    for path in result.result_files:
        print(path)
    if result.zip_path is not None:
        print(result.zip_path)
    return 0


def _read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        if not columns:
            raise ValueError("feature CSV has no header")
        if len(columns) != len(set(columns)):
            raise ValueError("feature CSV columns must be unique")
        rows = list(reader)
    if not rows:
        raise ValueError("feature CSV must contain at least one data row")
    return columns, rows


def _write_csv_rows(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_keypoints_by_video(path: Path) -> dict[str, list[PoseKeypoint]]:
    source = path.expanduser().resolve()
    if source.is_file():
        paths = (source,)
    elif source.is_dir():
        paths = tuple(
            candidate
            for candidate in sorted(source.rglob("*.csv"))
            if candidate.is_file() and not candidate.is_symlink()
        )
    else:
        raise FileNotFoundError(f"keypoint input does not exist: {source}")
    if not paths:
        raise ValueError(f"keypoint directory contains no CSV files: {source}")

    points_by_video: dict[str, list[PoseKeypoint]] = {}
    for csv_path in paths:
        for point in read_keypoints_csv(csv_path):
            points_by_video.setdefault(point.video_id, []).append(point)
    if not points_by_video:
        raise ValueError("keypoint input contains no data rows")
    return points_by_video


def _validated_scales(values: Iterable[float]) -> tuple[float, ...]:
    scales = tuple(float(value) for value in values)
    if not scales or any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("scales must be positive finite numbers")
    if len(scales) != len(set(scales)):
        raise ValueError("scales must be unique")
    return scales


def _validated_sample_rate(value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate_hz must be a positive finite number")
    return rate


def _format_number(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("multi-scale features must be finite")
    return format(number, ".17g")


if __name__ == "__main__":
    raise SystemExit(main())
