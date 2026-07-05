import numpy as np

from ssbd_behavior.features import (
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    multiscale_temporal_feature_dict,
)
from ssbd_behavior.pose import PoseKeypoint


def _moving_keypoints() -> list[PoseKeypoint]:
    rows = []
    sample_rate = 10.0
    for frame in range(40):
        timestamp = frame / sample_rate
        phase = 2.0 * np.pi * timestamp
        rows.append(
            PoseKeypoint(
                "synthetic-video",
                frame,
                timestamp,
                NOSE,
                0.5,
                0.4 + 0.03 * np.sin(phase),
                confidence=0.95,
            )
        )
        rows.append(
            PoseKeypoint(
                "synthetic-video",
                frame,
                timestamp,
                LEFT_WRIST,
                0.4 + 0.1 * np.sin(phase),
                0.6,
                confidence=0.9,
            )
        )
        if frame % 3:
            rows.append(
                PoseKeypoint(
                    "synthetic-video",
                    frame,
                    timestamp,
                    RIGHT_WRIST,
                    0.6 - 0.1 * np.sin(phase),
                    0.6,
                    confidence=0.85,
                )
            )
    return rows


def test_multiscale_features_are_stable_finite_and_quality_aware() -> None:
    features = multiscale_temporal_feature_dict(
        _moving_keypoints(),
        10.0,
        scales_s=(1.0, 2.0, 4.0),
        reference_end_s=4.0,
    )

    assert len(features) == 42
    assert all(np.isfinite(value) for value in features.values())
    assert all(
        any(name.startswith(prefix) for name in features)
        for prefix in ("ms_1s__", "ms_2s__", "ms_4s__")
    )
    assert features["ms_4s__motion_energy"] > 0.0
    assert features["ms_4s__wrist_distance_range"] > 0.0
    assert 0.0 < features["ms_4s__missingness_fraction"] < 1.0
    assert 0.0 <= features["ms_4s__left_right_symmetry"] <= 1.0


def test_multiscale_features_handle_completely_missing_keypoints() -> None:
    features = multiscale_temporal_feature_dict(
        [], 10.0, scales_s=(1.0,), reference_end_s=1.0
    )

    assert features["ms_1s__missingness_fraction"] == 1.0
    assert features["ms_1s__observed_frame_fraction"] == 0.0
    assert features["ms_1s__motion_energy"] == 0.0
    assert all(np.isfinite(value) for value in features.values())
