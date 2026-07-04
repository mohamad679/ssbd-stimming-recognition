import math

import pytest

from ssbd_behavior.features.engineering import (
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    dominant_frequency_hz,
    inter_landmark_distance_series,
    landmark_axis_series,
    periodicity_strength,
    window_feature_dict,
)
from ssbd_behavior.pose.keypoints import PoseKeypoint


def _row(
    frame: int,
    landmark: int,
    *,
    x: float,
    y: float,
    z: float = 0.0,
    sample_rate_hz: float = 20.0,
) -> PoseKeypoint:
    return PoseKeypoint(
        video_id="synthetic",
        frame_index=frame,
        timestamp_s=frame / sample_rate_hz,
        landmark_index=landmark,
        x=x,
        y=y,
        z=z,
        confidence=1.0,
    )


def test_dominant_frequency_detects_sine_wave() -> None:
    sample_rate = 40.0
    frequency = 4.0
    values = [
        math.sin(2.0 * math.pi * frequency * frame / sample_rate)
        for frame in range(80)
    ]

    assert dominant_frequency_hz(values, sample_rate) == pytest.approx(frequency)
    assert periodicity_strength(values, sample_rate) == pytest.approx(1.0)


def test_constant_signal_has_no_frequency_or_periodicity() -> None:
    values = [2.5] * 20

    assert dominant_frequency_hz(values, 20.0) == 0.0
    assert periodicity_strength(values, 20.0) == 0.0


def test_inter_landmark_distance_series_pairs_and_orders_frames() -> None:
    rows = [
        _row(1, RIGHT_WRIST, x=3.0, y=4.0),
        _row(0, LEFT_WRIST, x=0.0, y=0.0),
        _row(1, LEFT_WRIST, x=0.0, y=0.0),
        _row(0, RIGHT_WRIST, x=0.0, y=2.0),
    ]

    assert inter_landmark_distance_series(rows, LEFT_WRIST, RIGHT_WRIST) == (
        2.0,
        5.0,
    )
    assert landmark_axis_series(rows, LEFT_WRIST, "x") == (0.0, 0.0)


def test_window_feature_dict_has_stable_numeric_keys() -> None:
    sample_rate = 20.0
    rows = []
    for frame in range(40):
        wrist_x = math.sin(2.0 * math.pi * 2.0 * frame / sample_rate)
        head_y = math.sin(2.0 * math.pi * 1.0 * frame / sample_rate)
        rows.extend(
            [
                _row(frame, LEFT_WRIST, x=wrist_x, y=0.0),
                _row(frame, RIGHT_WRIST, x=wrist_x + 1.0, y=0.0),
                _row(frame, NOSE, x=0.0, y=head_y),
            ]
        )

    features = window_feature_dict(reversed(rows), sample_rate)

    assert tuple(features) == (
        "wrist_x_dominant_frequency_hz",
        "wrist_x_periodicity_strength",
        "inter_wrist_distance_mean",
        "inter_wrist_distance_std",
        "head_y_dominant_frequency_hz",
        "head_y_periodicity_strength",
    )
    assert all(type(value) is float and math.isfinite(value) for value in features.values())
    assert features["wrist_x_dominant_frequency_hz"] == pytest.approx(2.0)
    assert features["head_y_dominant_frequency_hz"] == pytest.approx(1.0)
    assert features["inter_wrist_distance_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("sample_rate", [0.0, -1.0, math.inf, math.nan, True])
def test_invalid_sample_rates_are_rejected(sample_rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        dominant_frequency_hz([0.0, 1.0, 0.0], sample_rate)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        periodicity_strength([0.0, 1.0, 0.0], sample_rate)


def test_invalid_axis_is_rejected() -> None:
    rows = [_row(0, LEFT_WRIST, x=0.0, y=0.0)]

    with pytest.raises(ValueError, match="axis must be"):
        landmark_axis_series(rows, LEFT_WRIST, "velocity")  # type: ignore[arg-type]
