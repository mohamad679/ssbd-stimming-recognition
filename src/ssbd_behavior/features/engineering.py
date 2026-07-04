"""Interpretable features computed from numeric pose-keypoint windows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite, sqrt
from numbers import Real
from typing import Literal

import numpy as np

from ssbd_behavior.pose.keypoints import PoseKeypoint

NOSE = 0
LEFT_WRIST = 15
RIGHT_WRIST = 16

Axis = Literal["x", "y", "z", "confidence"]
_VALID_AXES = frozenset(("x", "y", "z", "confidence"))


def dominant_frequency_hz(values: Iterable[float], sample_rate_hz: float) -> float:
    """Return the strongest non-DC FFT-bin frequency in hertz.

    Signals with fewer than three samples, or with no variation, return 0.0.
    """

    signal = _validated_signal(values)
    sample_rate = _validate_sample_rate(sample_rate_hz)
    if signal.size < 3 or np.ptp(signal) == 0.0:
        return 0.0

    centered = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 1:
        return 0.0
    power[0] = 0.0
    dominant_bin = int(np.argmax(power))
    if power[dominant_bin] == 0.0:
        return 0.0
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(frequencies[dominant_bin])


def periodicity_strength(values: Iterable[float], sample_rate_hz: float) -> float:
    """Return dominant non-DC power as a fraction of all non-DC FFT power.

    The result is between 0.0 and 1.0. Constant and short signals return 0.0.
    ``sample_rate_hz`` is validated for API consistency; the normalized power
    ratio itself is independent of its value.
    """

    signal = _validated_signal(values)
    _validate_sample_rate(sample_rate_hz)
    if signal.size < 3 or np.ptp(signal) == 0.0:
        return 0.0

    centered = signal - np.mean(signal)
    non_dc_power = np.abs(np.fft.rfft(centered))[1:] ** 2
    total_power = float(np.sum(non_dc_power))
    if total_power == 0.0:
        return 0.0
    return float(np.max(non_dc_power) / total_power)


def landmark_axis_series(
    rows: Iterable[PoseKeypoint], landmark_index: int, axis: Axis
) -> tuple[float, ...]:
    """Extract one landmark axis ordered by frame index, then timestamp."""

    if axis not in _VALID_AXES:
        raise ValueError(f"axis must be one of {sorted(_VALID_AXES)}")

    matching_rows = sorted(
        (row for row in rows if row.landmark_index == landmark_index),
        key=lambda row: (row.frame_index, row.timestamp_s),
    )
    values: list[float] = []
    for row in matching_rows:
        value = getattr(row, axis)
        if value is None:
            raise ValueError(f"axis {axis!r} is missing for landmark {landmark_index}")
        values.append(float(value))
    return tuple(values)


def inter_landmark_distance_series(
    rows: Iterable[PoseKeypoint], landmark_a: int, landmark_b: int
) -> tuple[float, ...]:
    """Return Euclidean distances for frames containing both landmarks.

    Distance is three-dimensional. A missing optional z coordinate is treated
    as zero, which also makes the function useful for explicitly 2-D pose rows.
    """

    if landmark_a == landmark_b:
        raise ValueError("landmark_a and landmark_b must be different")

    paired: dict[tuple[int, float], dict[int, PoseKeypoint]] = {}
    wanted = {landmark_a, landmark_b}
    for row in rows:
        if row.landmark_index not in wanted:
            continue
        frame_key = (row.frame_index, row.timestamp_s)
        frame = paired.setdefault(frame_key, {})
        if row.landmark_index in frame:
            raise ValueError(
                f"duplicate landmark {row.landmark_index} for frame {frame_key}"
            )
        frame[row.landmark_index] = row

    distances: list[float] = []
    for frame_key in sorted(paired):
        frame = paired[frame_key]
        if landmark_a not in frame or landmark_b not in frame:
            continue
        first = frame[landmark_a]
        second = frame[landmark_b]
        dz = (first.z or 0.0) - (second.z or 0.0)
        distances.append(
            sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + dz**2)
        )
    return tuple(distances)


def window_feature_dict(
    rows: Sequence[PoseKeypoint] | Iterable[PoseKeypoint], sample_rate_hz: float
) -> dict[str, float]:
    """Compute a stable flat feature mapping for one numeric keypoint window."""

    sample_rate = _validate_sample_rate(sample_rate_hz)
    window_rows = tuple(rows)
    wrist_x = landmark_axis_series(window_rows, LEFT_WRIST, "x")
    head_y = landmark_axis_series(window_rows, NOSE, "y")
    inter_wrist = inter_landmark_distance_series(
        window_rows, LEFT_WRIST, RIGHT_WRIST
    )
    distances = np.asarray(inter_wrist, dtype=float)

    return {
        "wrist_x_dominant_frequency_hz": dominant_frequency_hz(
            wrist_x, sample_rate
        ),
        "wrist_x_periodicity_strength": periodicity_strength(
            wrist_x, sample_rate
        ),
        "inter_wrist_distance_mean": (
            float(np.mean(distances)) if distances.size else 0.0
        ),
        "inter_wrist_distance_std": (
            float(np.std(distances)) if distances.size else 0.0
        ),
        "head_y_dominant_frequency_hz": dominant_frequency_hz(
            head_y, sample_rate
        ),
        "head_y_periodicity_strength": periodicity_strength(head_y, sample_rate),
    }


def _validate_sample_rate(sample_rate_hz: float) -> float:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, Real)
        or not isfinite(float(sample_rate_hz))
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be a positive finite number")
    return float(sample_rate_hz)


def _validated_signal(values: Iterable[float]) -> np.ndarray:
    try:
        signal = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must contain only numeric samples") from exc
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("values must be a one-dimensional finite numeric signal")
    return signal
