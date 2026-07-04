"""Typed, numeric representation of MediaPipe Pose keypoints."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real


MEDIAPIPE_POSE_LANDMARK_COUNT = 33

_SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_video_id(video_id: str) -> None:
    """Reject empty identifiers and values that could encode a path or URL."""

    if (
        not isinstance(video_id, str)
        or video_id in {".", ".."}
        or _SAFE_VIDEO_ID.fullmatch(video_id) is None
    ):
        raise ValueError(
            "video_id must be a non-empty identifier containing only letters, "
            "numbers, dots, underscores, or hyphens"
        )


def validate_landmark_index(landmark_index: int) -> None:
    """Validate a zero-based MediaPipe Pose landmark index."""

    if (
        isinstance(landmark_index, bool)
        or not isinstance(landmark_index, Integral)
        or not 0 <= landmark_index < MEDIAPIPE_POSE_LANDMARK_COUNT
    ):
        raise ValueError(
            f"landmark_index must be in [0, {MEDIAPIPE_POSE_LANDMARK_COUNT - 1}]"
        )


def validate_timestamp(timestamp_s: float) -> None:
    """Validate a finite, non-negative timestamp in seconds."""

    if not _is_finite_number(timestamp_s) or timestamp_s < 0:
        raise ValueError("timestamp_s must be a finite, non-negative number")


def validate_coordinate(value: float | None, name: str, *, optional: bool = False) -> None:
    """Validate a numeric coordinate, optionally permitting ``None``."""

    if optional and value is None:
        return
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite numeric coordinate")


def validate_confidence(confidence: float) -> None:
    """Validate landmark visibility/confidence on the closed unit interval."""

    if not _is_finite_number(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite number in [0, 1]")


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class PoseKeypoint:
    """One landmark observation from one video frame.

    The structure intentionally contains only a safe video identifier and numeric
    pose data. It cannot carry a source video path, image, or decoded frame.
    """

    video_id: str
    frame_index: int
    timestamp_s: float
    landmark_index: int
    x: float
    y: float
    z: float | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_video_id(self.video_id)
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, Integral)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        validate_timestamp(self.timestamp_s)
        validate_landmark_index(self.landmark_index)
        validate_coordinate(self.x, "x")
        validate_coordinate(self.y, "y")
        validate_coordinate(self.z, "z", optional=True)
        validate_confidence(self.confidence)
