"""Privacy-conscious, single-video pose-keypoint extraction.

Decoded frames exist only in memory.  The only persisted extraction artifact is
the numeric keypoint CSV written by :func:`write_keypoints_csv`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

from .io import write_keypoints_csv
from .keypoints import PoseKeypoint, validate_video_id


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SAFE_REPOSITORY_INPUT_ROOT = REPOSITORY_ROOT / "data" / "raw"
SAFE_REPOSITORY_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "processed"


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for extracting one local temporary video.

    Inputs inside the repository are rejected unless they are under the
    gitignored ``data/raw`` directory. Outputs inside the repository are
    restricted to the gitignored ``data/processed`` directory. The benchmark
    runner may additionally authorize its own raw and keypoint subdirectories
    through ``SSBD_BENCHMARK_WORK_ROOT``.
    """

    video_id: str
    input_video_path: Path | str
    output_csv_path: Path | str
    delete_input_after_success: bool = True
    max_frames: int | None = None

    def __post_init__(self) -> None:
        validate_video_id(self.video_id)
        if not isinstance(self.delete_input_after_success, bool):
            raise ValueError("delete_input_after_success must be a boolean")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, Integral)
            or self.max_frames < 0
        ):
            raise ValueError("max_frames must be a non-negative integer or None")

        input_path = Path(self.input_video_path).expanduser()
        output_path = Path(self.output_csv_path).expanduser()
        benchmark_work_root_text = os.environ.get("SSBD_BENCHMARK_WORK_ROOT")
        benchmark_work_root = (
            Path(benchmark_work_root_text).expanduser()
            if benchmark_work_root_text
            else None
        )
        benchmark_input_allowed = benchmark_work_root is not None and _is_within(
            input_path, benchmark_work_root / "temporary" / "raw"
        )
        benchmark_output_allowed = benchmark_work_root is not None and _is_within(
            output_path, benchmark_work_root / "artifacts" / "keypoints"
        )
        if not input_path.is_file():
            raise ValueError(f"input_video_path must be an existing local file: {input_path}")
        if output_path.suffix.lower() != ".csv":
            raise ValueError("output_csv_path must have a .csv extension")
        if (
            _is_within(input_path, REPOSITORY_ROOT)
            and not _is_within(input_path, SAFE_REPOSITORY_INPUT_ROOT)
            and not benchmark_input_allowed
        ):
            raise ValueError(
                "input_video_path must be outside the repository or under the "
                "gitignored data/raw directory (benchmark runs may use their "
                "designated temporary/raw directory)"
            )
        if (
            _is_within(output_path, REPOSITORY_ROOT)
            and not _is_within(output_path, SAFE_REPOSITORY_OUTPUT_ROOT)
            and not benchmark_output_allowed
        ):
            raise ValueError(
                "output_csv_path inside the repository must be under the "
                "gitignored data/processed directory (benchmark runs may use "
                "their designated artifacts/keypoints directory)"
            )
        if input_path.resolve() == output_path.resolve():
            raise ValueError("input_video_path and output_csv_path must differ")


@dataclass(frozen=True)
class FrameSample:
    """One in-memory decoded frame and its timestamp."""

    frame: Any
    timestamp_s: float


@dataclass(frozen=True)
class PoseLandmark:
    """Estimator output before video/frame metadata are attached."""

    landmark_index: int
    x: float
    y: float
    z: float | None = None
    confidence: float = 1.0


class PoseEstimator(Protocol):
    """Callable contract used by injected and optional MediaPipe estimators."""

    def __call__(self, frame: Any) -> Iterable[PoseLandmark] | None: ...


def extract_pose_keypoints(
    config: ExtractionConfig,
    frame_source: Iterable[FrameSample] | None = None,
    pose_estimator: PoseEstimator | None = None,
) -> list[PoseKeypoint]:
    """Extract and atomically persist numeric pose rows for one local video.

    Tests can inject an iterable of :class:`FrameSample` objects and a callable
    estimator. If either dependency is omitted, OpenCV/MediaPipe are imported
    lazily. The input is unlinked only after the completed CSV is in place.
    """

    if not isinstance(config, ExtractionConfig):
        raise TypeError("config must be an ExtractionConfig")

    owns_source = frame_source is None
    source = frame_source if frame_source is not None else _opencv_frame_source(
        Path(config.input_video_path)
    )
    owns_estimator = pose_estimator is None
    estimator = pose_estimator if pose_estimator is not None else _MediaPipePoseEstimator()
    output_path = Path(config.output_csv_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_csv_path(output_path)
    rows: list[PoseKeypoint] = []

    try:
        limited_source = (
            source if config.max_frames is None else islice(source, config.max_frames)
        )
        for frame_index, sample in enumerate(limited_source):
            if not isinstance(sample, FrameSample):
                raise TypeError("frame_source must yield FrameSample instances")
            landmarks = estimator(sample.frame)
            if landmarks is None:
                continue
            for landmark in landmarks:
                normalized = _coerce_landmark(landmark)
                rows.append(
                    PoseKeypoint(
                        video_id=config.video_id,
                        frame_index=frame_index,
                        timestamp_s=sample.timestamp_s,
                        landmark_index=normalized.landmark_index,
                        x=normalized.x,
                        y=normalized.y,
                        z=normalized.z,
                        confidence=normalized.confidence,
                    )
                )

        write_keypoints_csv(temporary_path, rows)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if owns_source:
            source.close()  # type: ignore[attr-defined]
        if owns_estimator:
            estimator.close()  # type: ignore[attr-defined]

    if config.delete_input_after_success:
        Path(config.input_video_path).unlink()
    return rows


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _temporary_csv_path(output_path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    return Path(name)


def _coerce_landmark(value: object) -> PoseLandmark:
    if isinstance(value, PoseLandmark):
        return value
    if isinstance(value, Mapping):
        return PoseLandmark(
            landmark_index=value["landmark_index"],  # type: ignore[arg-type]
            x=value["x"],  # type: ignore[arg-type]
            y=value["y"],  # type: ignore[arg-type]
            z=value.get("z"),  # type: ignore[arg-type]
            confidence=value.get("confidence", 1.0),  # type: ignore[arg-type]
        )
    raise TypeError("pose_estimator must return PoseLandmark objects or mappings")


def _opencv_frame_source(video_path: Path) -> Iterator[FrameSample]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotImplementedError(
            "Default video decoding requires optional dependency 'opencv-python'; "
            "inject frame_source in tests"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open local video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = (
                timestamp_ms / 1000.0
                if timestamp_ms >= 0
                else frame_index / fps if fps > 0 else float(frame_index)
            )
            yield FrameSample(frame=frame, timestamp_s=timestamp_s)
            frame_index += 1
    finally:
        capture.release()


class _MediaPipePoseEstimator:
    def __init__(self) -> None:
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NotImplementedError(
                "Default pose estimation requires optional dependency 'mediapipe'; "
                "inject pose_estimator in tests"
            ) from exc
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and getattr(solutions, "pose", None) is not None:
            self._backend = "solutions"
            self._pose = solutions.pose.Pose()
            return

        model_path = os.environ.get("MEDIAPIPE_POSE_LANDMARKER_TASK")
        if not model_path:
            raise NotImplementedError(
                "This MediaPipe build has no mp.solutions.pose. Set "
                "MEDIAPIPE_POSE_LANDMARKER_TASK to a temporary Pose Landmarker "
                ".task file to use the MediaPipe Tasks API."
            )
        candidate = Path(model_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                "MediaPipe Pose Landmarker task file does not exist: "
                f"{candidate}"
            )

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(candidate))
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
            )
            self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except (AttributeError, TypeError) as exc:
            raise NotImplementedError(
                "Installed 'mediapipe' exposes neither the legacy Pose solution "
                "nor a compatible Tasks PoseLandmarker API"
            ) from exc
        self._backend = "tasks"

    def __call__(self, frame: Any) -> list[PoseLandmark] | None:
        # OpenCV supplies BGR arrays; MediaPipe Pose expects RGB arrays.
        rgb_frame = frame[..., ::-1]
        if self._backend == "solutions":
            results = self._pose.process(rgb_frame)
            landmark_sets = (
                [] if results.pose_landmarks is None else [results.pose_landmarks.landmark]
            )
        else:
            import mediapipe as mp  # type: ignore[import-not-found]

            results = self._pose.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            )
            landmark_sets = results.pose_landmarks

        if not landmark_sets:
            return None
        landmarks = landmark_sets[0]
        return [
            PoseLandmark(
                landmark_index=index,
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                confidence=float(getattr(landmark, "visibility", 1.0) or 0.0),
            )
            for index, landmark in enumerate(landmarks)
        ]

    def close(self) -> None:
        self._pose.close()
