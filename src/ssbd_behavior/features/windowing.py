"""Deterministic fixed-duration windows and annotation-derived labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable

from ssbd_behavior.acquisition import SSBDPlusSegment
from ssbd_behavior.pose.keypoints import validate_video_id


_NEGATIVE_CATEGORIES = frozenset({"no-class", "no_class", "noclass"})


@dataclass(frozen=True)
class WindowSpec:
    """Duration and step size for fixed sliding windows."""

    window_size_s: float
    stride_s: float

    def __post_init__(self) -> None:
        _validate_positive_finite(self.window_size_s, "window_size_s")
        _validate_positive_finite(self.stride_s, "stride_s")


@dataclass(frozen=True)
class VideoWindow:
    """One complete, video-scoped time window and its binary label."""

    video_id: str
    start_s: float
    end_s: float
    label: int
    annotated_overlap_fraction: float


def generate_windows_for_video(
    video_id: str,
    duration_s: float,
    annotations: Iterable[SSBDPlusSegment],
    spec: WindowSpec,
    minimum_overlap_fraction: float = 0.5,
) -> list[VideoWindow]:
    """Generate complete windows for one video and label stereotypy overlap.

    Only annotations with the same ``video_id`` are considered, preventing an
    annotation from one video from affecting another video's windows. Overlapping
    positive annotations are merged before overlap is measured, so time is never
    double-counted.
    """

    validate_video_id(video_id)
    _validate_positive_finite(duration_s, "duration_s")
    if (
        not _is_finite_number(minimum_overlap_fraction)
        or not 0 <= minimum_overlap_fraction <= 1
    ):
        raise ValueError("minimum_overlap_fraction must be in [0, 1]")

    positive_intervals = _merge_intervals(
        (float(segment.start_time), float(segment.end_time))
        for segment in annotations
        if segment.video_id == video_id
        and segment.category.strip().lower() not in _NEGATIVE_CATEGORIES
    )

    windows: list[VideoWindow] = []
    window_index = 0
    while True:
        start_s = window_index * spec.stride_s
        end_s = start_s + spec.window_size_s
        if end_s > duration_s and not math.isclose(end_s, duration_s):
            break

        overlap_s = sum(
            max(0.0, min(end_s, interval_end) - max(start_s, interval_start))
            for interval_start, interval_end in positive_intervals
        )
        overlap_fraction = overlap_s / spec.window_size_s
        label = int(
            overlap_s > 0 and overlap_fraction >= minimum_overlap_fraction
        )
        windows.append(
            VideoWindow(
                video_id=video_id,
                start_s=start_s,
                end_s=end_s,
                label=label,
                annotated_overlap_fraction=overlap_fraction,
            )
        )
        window_index += 1

    return windows


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _validate_positive_finite(value: float, name: str) -> None:
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
