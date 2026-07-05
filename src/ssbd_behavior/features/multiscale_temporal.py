"""Multi-scale temporal features from privacy-safe numeric pose keypoints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite, sqrt
from numbers import Real

import numpy as np

from ssbd_behavior.features.engineering import (
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    dominant_frequency_hz,
    periodicity_strength,
)
from ssbd_behavior.pose.keypoints import PoseKeypoint


DEFAULT_TEMPORAL_SCALES_S = (1.0, 2.0, 4.0)
_RELEVANT_LANDMARKS = (NOSE, LEFT_WRIST, RIGHT_WRIST)


def multiscale_temporal_feature_dict(
    rows: Sequence[PoseKeypoint] | Iterable[PoseKeypoint],
    sample_rate_hz: float,
    *,
    scales_s: Iterable[float] = DEFAULT_TEMPORAL_SCALES_S,
    reference_end_s: float | None = None,
) -> dict[str, float]:
    """Compute aligned trailing-window features at each requested time scale.

    Every scale ends at ``reference_end_s``. When it is omitted, the end is one
    sample interval after the latest observation. Missing landmarks contribute
    to explicit quality features rather than causing imputation or failure.
    """

    sample_rate = _positive_finite(sample_rate_hz, "sample_rate_hz")
    scales = _validate_scales(scales_s)
    keypoints = tuple(rows)
    if any(not isinstance(row, PoseKeypoint) for row in keypoints):
        raise TypeError("rows must contain PoseKeypoint instances")

    if reference_end_s is None:
        end_s = (
            max((row.timestamp_s for row in keypoints), default=0.0) + 1.0 / sample_rate
        )
    else:
        end_s = _nonnegative_finite(reference_end_s, "reference_end_s")

    features: dict[str, float] = {}
    for scale_s in scales:
        start_s = end_s - scale_s
        scale_rows = tuple(
            row for row in keypoints if start_s <= row.timestamp_s < end_s
        )
        prefix = f"ms_{_format_scale(scale_s)}s__"
        for name, value in _single_scale_features(
            scale_rows, sample_rate=sample_rate, scale_s=scale_s
        ).items():
            features[prefix + name] = value
    return features


def _single_scale_features(
    rows: Sequence[PoseKeypoint], *, sample_rate: float, scale_s: float
) -> dict[str, float]:
    frames = _best_observations_by_frame(rows)
    wrist_distances = []
    for frame in frames.values():
        if LEFT_WRIST in frame and RIGHT_WRIST in frame:
            wrist_distances.append(_distance(frame[LEFT_WRIST], frame[RIGHT_WRIST]))

    left_speeds = _landmark_speeds(frames, LEFT_WRIST)
    right_speeds = _landmark_speeds(frames, RIGHT_WRIST)
    wrist_speeds = np.asarray(
        tuple(left_speeds.values()) + tuple(right_speeds.values()), dtype=float
    )
    wrist_acceleration = np.asarray(
        _speed_accelerations(left_speeds) + _speed_accelerations(right_speeds),
        dtype=float,
    )
    head_vertical_speed = np.asarray(
        tuple(_vertical_speeds(frames, NOSE).values()), dtype=float
    )
    distances = np.asarray(wrist_distances, dtype=float)
    periodic_signal = distances if distances.size else _axis_signal(frames, LEFT_WRIST)

    expected_frames = max(1, int(round(scale_s * sample_rate)))
    observed_slots = sum(
        landmark in frame
        for frame in frames.values()
        for landmark in _RELEVANT_LANDMARKS
    )
    expected_slots = expected_frames * len(_RELEVANT_LANDMARKS)
    confidences = np.asarray(
        [
            point.confidence
            for frame in frames.values()
            for point in frame.values()
            if point.landmark_index in _RELEVANT_LANDMARKS
        ],
        dtype=float,
    )
    motion_components = np.concatenate((wrist_speeds, head_vertical_speed))

    return {
        "wrist_distance_mean": _mean(distances),
        "wrist_distance_std": _std(distances),
        "wrist_distance_range": _range(distances),
        "wrist_velocity_mean": _mean(wrist_speeds),
        "wrist_velocity_std": _std(wrist_speeds),
        "wrist_acceleration_mean": _mean(wrist_acceleration),
        "head_vertical_motion_mean": _mean(head_vertical_speed),
        "dominant_frequency_hz": dominant_frequency_hz(periodic_signal, sample_rate),
        "periodicity_strength": periodicity_strength(periodic_signal, sample_rate),
        "left_right_symmetry": _symmetry(left_speeds, right_speeds),
        "pose_confidence_mean": _mean(confidences),
        "missingness_fraction": float(
            np.clip(1.0 - observed_slots / expected_slots, 0.0, 1.0)
        ),
        "observed_frame_fraction": float(
            np.clip(len(frames) / expected_frames, 0.0, 1.0)
        ),
        "motion_energy": (
            float(np.mean(motion_components**2)) if motion_components.size else 0.0
        ),
    }


def _best_observations_by_frame(
    rows: Sequence[PoseKeypoint],
) -> dict[tuple[int, float], dict[int, PoseKeypoint]]:
    frames: dict[tuple[int, float], dict[int, PoseKeypoint]] = {}
    for row in rows:
        if row.landmark_index not in _RELEVANT_LANDMARKS:
            continue
        frame = frames.setdefault((row.frame_index, row.timestamp_s), {})
        previous = frame.get(row.landmark_index)
        if previous is None or row.confidence > previous.confidence:
            frame[row.landmark_index] = row
    return dict(sorted(frames.items()))


def _landmark_speeds(
    frames: dict[tuple[int, float], dict[int, PoseKeypoint]], landmark: int
) -> dict[tuple[float, float], float]:
    points = [
        (timestamp, frame[landmark])
        for (_, timestamp), frame in frames.items()
        if landmark in frame
    ]
    speeds: dict[tuple[float, float], float] = {}
    for (first_t, first), (second_t, second) in zip(points, points[1:]):
        dt = second_t - first_t
        if dt > 0:
            speeds[(first_t, second_t)] = _distance(first, second) / dt
    return speeds


def _vertical_speeds(
    frames: dict[tuple[int, float], dict[int, PoseKeypoint]], landmark: int
) -> dict[tuple[float, float], float]:
    points = [
        (timestamp, frame[landmark])
        for (_, timestamp), frame in frames.items()
        if landmark in frame
    ]
    speeds: dict[tuple[float, float], float] = {}
    for (first_t, first), (second_t, second) in zip(points, points[1:]):
        dt = second_t - first_t
        if dt > 0:
            speeds[(first_t, second_t)] = abs(second.y - first.y) / dt
    return speeds


def _speed_accelerations(speeds: dict[tuple[float, float], float]) -> tuple[float, ...]:
    samples = tuple(speeds.items())
    accelerations = []
    for ((first_start, first_end), first), ((second_start, second_end), second) in zip(
        samples, samples[1:]
    ):
        first_midpoint = (first_start + first_end) / 2.0
        second_midpoint = (second_start + second_end) / 2.0
        dt = second_midpoint - first_midpoint
        if dt > 0:
            accelerations.append(abs(second - first) / dt)
    return tuple(accelerations)


def _axis_signal(
    frames: dict[tuple[int, float], dict[int, PoseKeypoint]], landmark: int
) -> np.ndarray:
    return np.asarray(
        [frame[landmark].x for frame in frames.values() if landmark in frame],
        dtype=float,
    )


def _symmetry(
    left: dict[tuple[float, float], float],
    right: dict[tuple[float, float], float],
) -> float:
    common = tuple(sorted(set(left) & set(right)))
    if not common:
        return 0.0
    difference = np.mean([abs(left[key] - right[key]) for key in common])
    magnitude = np.mean([abs(left[key]) + abs(right[key]) for key in common])
    if magnitude == 0.0:
        return 1.0
    return float(np.clip(1.0 - difference / magnitude, 0.0, 1.0))


def _distance(first: PoseKeypoint, second: PoseKeypoint) -> float:
    dz = (first.z or 0.0) - (second.z or 0.0)
    return sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + dz**2)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _range(values: np.ndarray) -> float:
    return float(np.ptp(values)) if values.size else 0.0


def _validate_scales(scales_s: Iterable[float]) -> tuple[float, ...]:
    scales = tuple(_positive_finite(scale, "scale") for scale in scales_s)
    if not scales:
        raise ValueError("scales_s must contain at least one scale")
    if len(scales) != len(set(scales)):
        raise ValueError("scales_s must not contain duplicates")
    return scales


def _positive_finite(value: float, name: str) -> float:
    result = _nonnegative_finite(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _format_scale(scale_s: float) -> str:
    return f"{scale_s:g}".replace(".", "p")
