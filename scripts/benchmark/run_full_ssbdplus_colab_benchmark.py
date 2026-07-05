#!/usr/bin/env python3
"""Resume-safe, privacy-preserving SSBD+ benchmark orchestration for Colab."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence
from urllib.request import urlopen
import zipfile


EXPECTED_SEGMENT_COUNT = 65
EXPECTED_VIDEO_COUNT = 36
DOWNLOAD_MARGIN_S = 5.0
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
METADATA_COLUMNS = (
    "xml_file_name",
    "youtube_video_url",
    "action_start_time",
    "action_end_time",
    "action_category",
)
KEYPOINT_COLUMNS = (
    "video_id",
    "frame_index",
    "timestamp_s",
    "landmark_index",
    "x",
    "y",
    "z",
    "confidence",
)
FEATURE_BASE_COLUMNS = ("video_id", "window_start_s", "window_end_s", "label")
SAFE_ARCHIVE_SUFFIXES = frozenset({".csv", ".json", ".svg", ".txt"})
FORBIDDEN_ARCHIVE_SUFFIXES = frozenset(
    {
        ".avi", ".bmp", ".gif", ".jpeg", ".jpg", ".m4v", ".mkv", ".mov",
        ".mp4", ".png", ".task", ".webm", ".webp", ".joblib", ".onnx",
        ".pickle", ".pkl", ".pt", ".pth", ".safetensors",
    }
)
_SAFE_FILE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    segment_id: int
    video_id: str
    url: str
    start_s: float
    end_s: float
    action_category: str
    annotation_file: str


@dataclass(frozen=True, slots=True)
class VideoRecord:
    video_id: str
    url: str
    max_end_s: float
    action_categories: tuple[str, ...]
    primary_action_category: str
    segment_count: int


@dataclass(frozen=True, slots=True)
class MetadataBundle:
    csv_path: Path
    segments: tuple[SegmentRecord, ...]
    videos: tuple[VideoRecord, ...]
    category_codes: dict[str, int]


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    artifacts: Path
    keypoints: Path
    features: Path
    reports: Path
    manifests: Path
    svgs: Path
    temporary: Path
    raw: Path
    runtime: Path
    state: Path


class StageError(RuntimeError):
    """A recoverable per-stage failure."""


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run the full SSBD+ benchmark with temporary video handling, grouped "
            "validation, resumable numeric artifacts, and a strictly safe zip."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform downloads and write outputs (default is validation only)",
    )
    return parser


def validate_metadata(
    metadata_dir: Path | str,
    *,
    expected_segment_count: int | None = None,
    expected_video_count: int | None = None,
) -> MetadataBundle:
    """Load and validate segment-level metadata without collapsing categories."""

    csv_path = locate_metadata_csv(Path(metadata_dir).expanduser())
    segments: list[SegmentRecord] = []
    urls_by_video: dict[str, str] = {}
    seen_segments: set[tuple[str, float, float, str]] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        missing = [name for name in METADATA_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"metadata is missing required columns: {', '.join(missing)}")
        if len(columns) != len(set(columns)):
            raise ValueError("metadata columns must be unique")

        for row_number, row in enumerate(reader, start=2):
            try:
                annotation_file = _required_text(row["xml_file_name"], "xml_file_name")
                video_id = Path(annotation_file).stem.strip()
                if not video_id or video_id in {".", ".."}:
                    raise ValueError("xml_file_name must yield a non-empty video id")
                url = _required_text(row["youtube_video_url"], "youtube_video_url")
                if not url.startswith(("https://", "http://")):
                    raise ValueError("youtube_video_url must be an HTTP(S) URL")
                start_s = float(row["action_start_time"])
                end_s = float(row["action_end_time"])
                category = _required_text(row["action_category"], "action_category")
                if not math.isfinite(start_s) or not math.isfinite(end_s):
                    raise ValueError("segment times must be finite")
                if start_s < 0 or start_s >= end_s:
                    raise ValueError("segment times must satisfy 0 <= start < end")
                prior_url = urls_by_video.setdefault(video_id, url)
                if prior_url != url:
                    raise ValueError(f"video {video_id!r} has conflicting URLs")
                identity = (video_id, start_s, end_s, category)
                if identity in seen_segments:
                    raise ValueError("duplicate segment annotation")
                seen_segments.add(identity)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid metadata row {row_number}: {exc}") from exc
            segments.append(
                SegmentRecord(
                    segment_id=len(segments) + 1,
                    video_id=video_id,
                    url=url,
                    start_s=start_s,
                    end_s=end_s,
                    action_category=category,
                    annotation_file=annotation_file,
                )
            )

    if not segments:
        raise ValueError("metadata must contain at least one segment")
    videos = build_video_records(segments)
    if expected_segment_count is not None and len(segments) != expected_segment_count:
        raise ValueError(
            f"expected {expected_segment_count} segment rows, found {len(segments)}"
        )
    if expected_video_count is not None and len(videos) != expected_video_count:
        raise ValueError(f"expected {expected_video_count} unique videos, found {len(videos)}")
    category_codes = {
        category: index
        for index, category in enumerate(
            sorted({segment.action_category for segment in segments})
        )
    }
    return MetadataBundle(csv_path, tuple(segments), videos, category_codes)


def locate_metadata_csv(path: Path) -> Path:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*.csv"))
    else:
        raise FileNotFoundError(f"metadata path does not exist: {path}")

    matching: list[Path] = []
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8-sig", newline="") as stream:
                columns = set(next(csv.reader(stream), ()))
        except OSError:
            continue
        if set(METADATA_COLUMNS).issubset(columns):
            matching.append(candidate)
    if not matching:
        raise FileNotFoundError(
            f"no CSV with the required SSBD+ metadata columns was found under {path}"
        )
    if len(matching) > 1:
        names = ", ".join(str(candidate) for candidate in matching)
        raise ValueError(f"multiple SSBD+ metadata CSV files found; provide one file: {names}")
    return matching[0]


def build_video_records(segments: Iterable[SegmentRecord]) -> tuple[VideoRecord, ...]:
    grouped: dict[str, list[SegmentRecord]] = {}
    for segment in segments:
        grouped.setdefault(segment.video_id, []).append(segment)
    records: list[VideoRecord] = []
    for video_id in sorted(grouped):
        video_segments = grouped[video_id]
        urls = {segment.url for segment in video_segments}
        if len(urls) != 1:
            raise ValueError(f"video {video_id!r} has conflicting URLs")
        durations: dict[str, float] = {}
        for segment in video_segments:
            durations[segment.action_category] = durations.get(
                segment.action_category, 0.0
            ) + (segment.end_s - segment.start_s)
        categories = tuple(sorted(durations))
        primary = sorted(categories, key=lambda name: (-durations[name], name))[0]
        records.append(
            VideoRecord(
                video_id=video_id,
                url=next(iter(urls)),
                max_end_s=max(segment.end_s for segment in video_segments),
                action_categories=categories,
                primary_action_category=primary,
                segment_count=len(video_segments),
            )
        )
    return tuple(records)


def should_skip_completed_output(
    path: Path | str,
    *,
    resume: bool,
    required_columns: Sequence[str],
    expected_video_id: str | None = None,
) -> bool:
    """Return true only for a structurally valid, non-empty resumable CSV."""

    if not resume:
        return False
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        with candidate.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            if not all(name in columns for name in required_columns):
                return False
            if tuple(required_columns) == KEYPOINT_COLUMNS and columns != KEYPOINT_COLUMNS:
                return False
            row_count = 0
            for row in reader:
                row_count += 1
                if expected_video_id is not None and row.get("video_id") != expected_video_id:
                    return False
                if tuple(required_columns) == KEYPOINT_COLUMNS:
                    _validate_keypoint_csv_row(row)
                elif tuple(required_columns) == FEATURE_BASE_COLUMNS:
                    _validate_feature_csv_row(row, columns)
            return row_count > 0
    except (KeyError, OSError, TypeError, ValueError, csv.Error):
        return False


def _validate_keypoint_csv_row(row: dict[str, str]) -> None:
    if not row["video_id"].strip():
        raise ValueError("empty video_id")
    if int(row["frame_index"]) < 0 or int(row["landmark_index"]) < 0:
        raise ValueError("negative keypoint index")
    values = [
        float(row["timestamp_s"]),
        float(row["x"]),
        float(row["y"]),
        float(row["confidence"]),
    ]
    if row["z"] != "":
        values.append(float(row["z"]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite keypoint value")


def _validate_feature_csv_row(row: dict[str, str], columns: Sequence[str]) -> None:
    if not row["video_id"].strip():
        raise ValueError("empty video_id")
    start_s = float(row["window_start_s"])
    end_s = float(row["window_end_s"])
    label = int(row["label"])
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        raise ValueError("invalid feature window")
    if label not in (0, 1) or str(label) != row["label"]:
        raise ValueError("invalid feature label")
    feature_values = [float(row[name]) for name in columns if name not in FEATURE_BASE_COLUMNS]
    if not feature_values or not all(math.isfinite(value) for value in feature_values):
        raise ValueError("invalid numeric feature values")


def create_safe_zip(artifact_root: Path | str, output_zip: Path | str) -> tuple[str, ...]:
    """Archive only allowlisted artifacts and reject any unsafe source member."""

    root = Path(artifact_root).resolve()
    destination = Path(output_zip).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"artifact root does not exist: {root}")
    if destination == root or root in destination.parents:
        raise ValueError("output zip must be outside the artifact root")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("artifact root contains no files")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"symbolic links are not allowed in the final archive: {path}")
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
            raise ValueError(f"forbidden media/image/model artifact: {path}")
        if suffix not in SAFE_ARCHIVE_SUFFIXES:
            raise ValueError(f"artifact extension is not allowlisted: {path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    members: list[str] = []
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                member = path.relative_to(root).as_posix()
                archive.write(path, member)
                members.append(member)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return tuple(members)


def prepare_run_paths(work_dir: Path) -> RunPaths:
    root = work_dir.expanduser().resolve()
    artifacts = root / "artifacts"
    paths = RunPaths(
        root=root,
        artifacts=artifacts,
        keypoints=artifacts / "keypoints",
        features=artifacts / "features",
        reports=artifacts / "reports",
        manifests=artifacts / "manifests",
        svgs=artifacts / "svgs",
        temporary=root / "temporary",
        raw=root / "temporary" / "raw",
        runtime=root / "temporary" / "runtime",
        state=root / "run_state.json",
    )
    for directory in (
        paths.keypoints, paths.features, paths.reports, paths.manifests,
        paths.svgs, paths.raw, paths.runtime,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_runtime_dependencies(repo_root: Path) -> None:
    missing: list[str] = []
    for module, package in (
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("cv2", "opencv-python-headless"),
        ("mediapipe", "mediapipe"),
        ("yt_dlp", "yt-dlp"),
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(package)
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (system executable)")
    required_scripts = (
        "scripts/pose/extract_keypoints.py",
        "scripts/features/build_feature_table.py",
        "scripts/run_baselines.py",
        "scripts/run_loso.py",
        "scripts/run_permutation_test.py",
        "scripts/explain_feature_importance.py",
        "scripts/render_skeleton_svg.py",
    )
    for relative in required_scripts:
        if not (repo_root / relative).is_file():
            missing.append(relative)
    if missing:
        raise RuntimeError(
            "missing benchmark dependencies: " + ", ".join(missing) + ". "
            "Install the project benchmark extras and ensure ffmpeg is on PATH."
        )


def mediapipe_requires_task_model() -> bool:
    import mediapipe as mp  # type: ignore[import-not-found]

    solutions = getattr(mp, "solutions", None)
    return solutions is None or getattr(solutions, "pose", None) is None


def download_task_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".task.tmp")
    try:
        with urlopen(TASK_MODEL_URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if temporary.stat().st_size < 1024:
            raise RuntimeError("downloaded MediaPipe task model is unexpectedly small")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_benchmark(args: argparse.Namespace, metadata: MetadataBundle) -> int:
    repo_root = args.repo_root.expanduser().resolve()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1")
    if args.max_height < 1:
        raise ValueError("--max-height must be positive")
    if not args.execute:
        print(
            f"Validated {len(metadata.segments)} segments across "
            f"{len(metadata.videos)} videos. Dry run only; use --execute to run."
        )
        return 0

    ensure_runtime_dependencies(repo_root)
    paths = prepare_run_paths(args.work_dir)
    if not args.resume:
        _cleanup_directory(paths.artifacts)
        for directory in (
            paths.keypoints, paths.features, paths.reports, paths.manifests, paths.svgs
        ):
            directory.mkdir(parents=True, exist_ok=True)
    _cleanup_directory(paths.raw)
    state = _read_state(paths.state) if args.resume else {"schema_version": 1, "videos": {}}
    metadata_sha256 = _sha256(metadata.csv_path)
    previous_metadata_sha256 = state.get("metadata_sha256")
    if args.resume and previous_metadata_sha256 not in {None, metadata_sha256}:
        raise ValueError(
            "resume state belongs to different metadata; use a new work directory "
            "or run without --resume"
        )
    state["metadata_sha256"] = metadata_sha256
    _write_json_atomic(paths.state, state)
    failures: list[dict[str, str]] = []
    task_model = paths.runtime / "pose_landmarker_lite.task"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    environment["SSBD_BENCHMARK_WORK_ROOT"] = str(paths.root)

    try:
        if mediapipe_requires_task_model():
            if not task_model.is_file():
                download_task_model(task_model)
            environment["MEDIAPIPE_POSE_LANDMARKER_TASK"] = str(task_model)
        _write_metadata_artifacts(paths, metadata)

        for video in metadata.videos:
            safe_name = _safe_video_filename(video.video_id)
            keypoint_csv = paths.keypoints / f"{safe_name}.csv"
            feature_csv = paths.features / "per_video" / f"{safe_name}.csv"
            feature_csv.parent.mkdir(parents=True, exist_ok=True)
            video_state = state.setdefault("videos", {}).setdefault(video.video_id, {})

            keypoints_ready = should_skip_completed_output(
                keypoint_csv,
                resume=args.resume,
                required_columns=KEYPOINT_COLUMNS,
                expected_video_id=video.video_id,
            )
            if not keypoints_ready:
                raw_path: Path | None = None
                try:
                    raw_path = _download_video(video, paths.raw, args.max_height, environment)
                    _run_command(
                        [
                            sys.executable,
                            str(repo_root / "scripts/pose/extract_keypoints.py"),
                            video.video_id,
                            str(raw_path),
                            str(keypoint_csv),
                            "--execute",
                        ],
                        repo_root,
                        environment,
                    )
                    if not should_skip_completed_output(
                        keypoint_csv,
                        resume=True,
                        required_columns=KEYPOINT_COLUMNS,
                        expected_video_id=video.video_id,
                    ):
                        raise StageError("pose extraction produced no keypoint rows")
                    video_state["keypoints"] = "complete"
                    _write_json_atomic(paths.state, state)
                    keypoints_ready = True
                except Exception as exc:
                    _record_failure(failures, video.video_id, "keypoint_extraction", exc)
                    video_state["keypoints"] = "failed"
                    _write_json_atomic(paths.state, state)
                finally:
                    if raw_path is not None:
                        raw_path.unlink(missing_ok=True)
                    _delete_download_candidates(paths.raw, safe_name)

            if not keypoints_ready:
                continue

            features_ready = should_skip_completed_output(
                feature_csv,
                resume=args.resume,
                required_columns=FEATURE_BASE_COLUMNS,
                expected_video_id=video.video_id,
            )
            if not features_ready:
                try:
                    sample_rate = infer_sample_rate_hz(keypoint_csv)
                    _run_command(
                        [
                            sys.executable,
                            str(repo_root / "scripts/features/build_feature_table.py"),
                            str(keypoint_csv),
                            str(feature_csv),
                            "--annotations-csv",
                            str(metadata.csv_path),
                            "--sample-rate-hz",
                            f"{sample_rate:.12g}",
                            "--execute",
                        ],
                        repo_root,
                        environment,
                    )
                    if not should_skip_completed_output(
                        feature_csv,
                        resume=True,
                        required_columns=FEATURE_BASE_COLUMNS,
                        expected_video_id=video.video_id,
                    ):
                        raise StageError("feature generation produced no complete windows")
                    video_state["features"] = "complete"
                    _write_json_atomic(paths.state, state)
                except Exception as exc:
                    _record_failure(failures, video.video_id, "feature_generation", exc)
                    video_state["features"] = "failed"
                    _write_json_atomic(paths.state, state)

            svg_path = paths.svgs / f"{safe_name}.svg"
            if not (args.resume and svg_path.is_file()):
                try:
                    _run_command(
                        [
                            sys.executable,
                            str(repo_root / "scripts/render_skeleton_svg.py"),
                            str(keypoint_csv),
                            "--output",
                            str(svg_path),
                            "--execute",
                        ],
                        repo_root,
                        environment,
                    )
                except Exception as exc:
                    _record_failure(failures, video.video_id, "skeleton_svg", exc)

        merged_feature_csv = paths.features / "all_features.csv"
        feature_inputs = [
            paths.features / "per_video" / f"{_safe_video_filename(video.video_id)}.csv"
            for video in metadata.videos
        ]
        valid_features = [
            path for path in feature_inputs
            if should_skip_completed_output(path, resume=True, required_columns=FEATURE_BASE_COLUMNS)
        ]
        evaluation_completed = False
        if valid_features:
            try:
                merge_csv_files(valid_features, merged_feature_csv)
                evaluation_failure_count = len(failures)
                _run_evaluations(
                    repo_root, paths, merged_feature_csv, args.n_splits,
                    args.n_permutations, environment, failures,
                )
                evaluation_completed = len(failures) == evaluation_failure_count
            except Exception as exc:
                _record_failure(failures, "", "evaluation", exc)
        else:
            _record_failure(failures, "", "evaluation", StageError("no feature tables available"))

        _write_failure_report(paths.artifacts / "failure_report.csv", failures)
        completed_videos = sum(
            should_skip_completed_output(path, resume=True, required_columns=FEATURE_BASE_COLUMNS)
            for path in feature_inputs
        )
        summary = {
            "schema_version": 1,
            "status": "completed" if not failures else "completed_with_failures",
            "generated_utc": _utc_now(),
            "segment_count": len(metadata.segments),
            "video_count": len(metadata.videos),
            "completed_video_feature_tables": completed_videos,
            "failure_count": len(failures),
            "n_splits": args.n_splits,
            "n_permutations": args.n_permutations,
            "max_height": args.max_height,
            "resume": bool(args.resume),
            "privacy": {
                "raw_videos_persisted": False,
                "frames_or_images_persisted": False,
                "trained_models_persisted": False,
            },
        }
        _write_json(paths.artifacts / "run_summary.json", summary)
        _write_artifact_manifest(paths.artifacts)
        create_safe_zip(paths.artifacts, args.output_zip)
        return benchmark_exit_code(
            feature_table_count=len(valid_features),
            evaluation_completed=evaluation_completed,
            safe_zip_created=args.output_zip.expanduser().resolve().is_file(),
        )
    finally:
        _cleanup_directory(paths.raw)
        task_model.unlink(missing_ok=True)


def benchmark_exit_code(
    *,
    feature_table_count: int,
    evaluation_completed: bool,
    safe_zip_created: bool,
) -> int:
    """Return success only when all fatal benchmark requirements are satisfied."""

    return int(
        feature_table_count < 1
        or not evaluation_completed
        or not safe_zip_created
    )


def infer_sample_rate_hz(keypoint_csv: Path) -> float:
    timestamps_by_frame: dict[int, float] = {}
    with keypoint_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            timestamps_by_frame.setdefault(int(row["frame_index"]), float(row["timestamp_s"]))
    timestamps = [timestamps_by_frame[index] for index in sorted(timestamps_by_frame)]
    differences = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    if not differences:
        raise ValueError("at least two timestamped pose frames are required")
    interval = statistics.median(differences)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("could not infer a positive sample interval")
    return 1.0 / interval


def merge_csv_files(inputs: Sequence[Path], output: Path) -> None:
    header: tuple[str, ...] | None = None
    rows: list[dict[str, str]] = []
    for path in inputs:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            current_header = tuple(reader.fieldnames or ())
            if header is None:
                header = current_header
            elif current_header != header:
                raise ValueError(f"feature CSV schema mismatch: {path}")
            rows.extend(reader)
    if not header or not rows:
        raise ValueError("cannot merge empty feature tables")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def _run_evaluations(
    repo_root: Path,
    paths: RunPaths,
    feature_csv: Path,
    n_splits: int,
    n_permutations: int,
    environment: dict[str, str],
    failures: list[dict[str, str]],
) -> None:
    commands = (
        (
            "group_kfold",
            [sys.executable, str(repo_root / "scripts/run_baselines.py"), str(feature_csv), "--n-splits", str(n_splits)],
        ),
        (
            "loso",
            [sys.executable, str(repo_root / "scripts/run_loso.py"), str(feature_csv)],
        ),
        (
            "permutation",
            [
                sys.executable, str(repo_root / "scripts/run_permutation_test.py"),
                str(feature_csv), "--n-splits", str(n_splits),
                "--n-permutations", str(n_permutations),
            ],
        ),
    )
    for name, command in commands:
        report = paths.reports / f"{name}.txt"
        try:
            _run_command_report(command, repo_root, environment, report)
        except Exception as exc:
            _record_failure(failures, "", name, exc)
    importance = paths.reports / "feature_importance.json"
    try:
        _run_command(
            [
                sys.executable, str(repo_root / "scripts/explain_feature_importance.py"),
                str(feature_csv), "--model", "logistic_regression", "--top-k", "20",
                "--output", str(importance), "--execute",
            ],
            repo_root,
            environment,
        )
    except Exception as exc:
        _record_failure(failures, "", "feature_importance", exc)


def _download_video(
    video: VideoRecord,
    raw_dir: Path,
    max_height: int,
    environment: dict[str, str],
) -> Path:
    safe_name = _safe_video_filename(video.video_id)
    _delete_download_candidates(raw_dir, safe_name)
    output_template = raw_dir / f"{safe_name}.%(ext)s"
    cutoff = video.max_end_s + DOWNLOAD_MARGIN_S
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-part",
        "--restrict-filenames",
        "--format",
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]/best[height<={max_height}]",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        f"*0-{cutoff:.3f}",
        "--force-keyframes-at-cuts",
        "--output",
        str(output_template),
        video.url,
    ]
    _run_command(command, raw_dir, environment)
    candidates = sorted(
        path for path in raw_dir.glob(f"{safe_name}.*")
        if path.is_file() and path.suffix.lower() not in {".part", ".ytdl"}
    )
    if len(candidates) != 1:
        raise StageError(
            f"yt-dlp produced {len(candidates)} candidate video files for {video.video_id}"
        )
    return candidates[0]


def _write_metadata_artifacts(paths: RunPaths, metadata: MetadataBundle) -> None:
    segment_path = paths.features / "segment_labels.csv"
    with segment_path.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "segment_id", "video_id", "action_start_s", "action_end_s",
            "action_category_code",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for segment in metadata.segments:
            writer.writerow(
                {
                    "segment_id": segment.segment_id,
                    "video_id": segment.video_id,
                    "action_start_s": segment.start_s,
                    "action_end_s": segment.end_s,
                    "action_category_code": metadata.category_codes[segment.action_category],
                }
            )
    _write_json(
        paths.manifests / "metadata_manifest.json",
        {
            "schema_version": 1,
            "segment_count": len(metadata.segments),
            "video_count": len(metadata.videos),
            "action_category_codes": metadata.category_codes,
            "videos": [
                {
                    "video_id": video.video_id,
                    "segment_count": video.segment_count,
                    "action_categories": list(video.action_categories),
                    "primary_action_category": video.primary_action_category,
                    "primary_action_category_usage": "reporting_only",
                }
                for video in metadata.videos
            ],
        },
    )


def _write_artifact_manifest(artifact_root: Path) -> None:
    destination = artifact_root / "manifests" / "artifact_manifest.json"
    records = []
    for path in sorted(candidate for candidate in artifact_root.rglob("*") if candidate.is_file()):
        if path == destination:
            continue
        suffix = path.suffix.lower()
        if suffix not in SAFE_ARCHIVE_SUFFIXES or suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
            raise ValueError(f"unsafe artifact cannot be manifested: {path}")
        records.append(
            {
                "path": path.relative_to(artifact_root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "artifact_type": path.relative_to(artifact_root).parts[0],
            }
        )
    _write_json(
        destination,
        {"schema_version": 1, "generated_utc": _utc_now(), "records": records},
    )


def _run_command(
    command: Sequence[str], cwd: Path, environment: dict[str, str]
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise StageError(f"command failed ({result.returncode}): {detail[-2000:]}")
    return result.stdout


def _run_command_report(
    command: Sequence[str], cwd: Path, environment: dict[str, str], report: Path
) -> None:
    try:
        output = _run_command(command, cwd, environment)
    except Exception as exc:
        report.write_text(f"status: failed\nerror: {exc}\n", encoding="utf-8")
        raise
    report.write_text(output, encoding="utf-8")


def _write_failure_report(path: Path, failures: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("video_id", "stage", "error"))
        writer.writeheader()
        writer.writerows(failures)


def _record_failure(
    failures: list[dict[str, str]], video_id: str, stage: str, error: Exception
) -> None:
    message = str(error).replace("\n", " ")
    message = re.sub(r"https?://\S+", "[url omitted]", message)
    failures.append(
        {"video_id": video_id, "stage": stage, "error": message[:2000]}
    )


def _read_state(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "videos": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "videos": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("videos"), dict):
        return {"schema_version": 1, "videos": {}}
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    _write_json_atomic(path, payload)


def _cleanup_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            path.rmdir()


def _delete_download_candidates(raw_dir: Path, safe_name: str) -> None:
    for candidate in raw_dir.glob(f"{safe_name}.*"):
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink(missing_ok=True)


def _safe_video_filename(video_id: str) -> str:
    component = _SAFE_FILE_COMPONENT.sub("_", video_id).strip("._") or "video"
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:10]
    return f"{component[:80]}-{digest}"


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = validate_metadata(
            args.metadata_dir,
            expected_segment_count=EXPECTED_SEGMENT_COUNT,
            expected_video_count=EXPECTED_VIDEO_COUNT,
        )
        return run_benchmark(args, metadata)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
