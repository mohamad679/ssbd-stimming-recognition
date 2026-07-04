import pytest

from ssbd_behavior.pose import (
    MEDIAPIPE_POSE_LANDMARK_COUNT,
    PoseKeypoint,
    read_keypoints_csv,
    write_keypoints_csv,
)


def test_keypoint_csv_roundtrip(tmp_path):
    rows = [
        PoseKeypoint("video-1", 0, 0.0, 0, 0.25, 0.5, -0.1, 0.95),
        PoseKeypoint("video-1", 0, 0.0, 1, 0.4, 0.6, None, 0.8),
        PoseKeypoint("video-1", 1, 1 / 30, 0, 0.3, 0.55, -0.08, 1.0),
    ]
    csv_path = tmp_path / "keypoints.csv"

    write_keypoints_csv(csv_path, rows)

    assert read_keypoints_csv(csv_path) == rows
    assert "video_path" not in csv_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("landmark_index", [-1, MEDIAPIPE_POSE_LANDMARK_COUNT])
def test_invalid_landmark_index_is_rejected(landmark_index):
    with pytest.raises(ValueError, match="landmark_index"):
        PoseKeypoint("video-1", 0, 0.0, landmark_index, 0.2, 0.3, confidence=0.9)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError, match="confidence"):
        PoseKeypoint("video-1", 0, 0.0, 0, 0.2, 0.3, confidence=confidence)


@pytest.mark.parametrize(
    ("timestamp_s", "x", "y"),
    [(-0.1, 0.2, 0.3), (0.0, None, 0.3), (0.0, 0.2, None)],
)
def test_negative_timestamp_and_missing_coordinates_are_rejected(timestamp_s, x, y):
    with pytest.raises(ValueError):
        PoseKeypoint("video-1", 0, timestamp_s, 0, x, y, confidence=0.9)
