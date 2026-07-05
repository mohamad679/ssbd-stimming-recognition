from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from ssbd_behavior.pose import (
    ExtractionConfig,
    FrameSample,
    PoseLandmark,
    extract_pose_keypoints,
    read_keypoints_csv,
)
from ssbd_behavior.pose.extract_keypoints import _MediaPipePoseEstimator


def _config(tmp_path, *, delete=True, max_frames=None):
    input_path = tmp_path / "temporary-video.mp4"
    input_path.write_bytes(b"synthetic test placeholder; never decoded")
    return ExtractionConfig(
        video_id="synthetic-1",
        input_video_path=input_path,
        output_csv_path=tmp_path / "keypoints.csv",
        delete_input_after_success=delete,
        max_frames=max_frames,
    )


def _frames(count=3):
    return [FrameSample(frame={"value": index}, timestamp_s=index / 10) for index in range(count)]


def _estimator(frame):
    return [PoseLandmark(0, frame["value"] / 10, 0.5, -0.1, 0.9)]


def test_successful_extraction_writes_numeric_csv(tmp_path):
    config = _config(tmp_path, delete=False)

    result = extract_pose_keypoints(config, _frames(), _estimator)

    assert result == read_keypoints_csv(config.output_csv_path)
    assert len(result) == 3
    csv_text = Path(config.output_csv_path).read_text(encoding="utf-8")
    assert "temporary-video" not in csv_text
    assert "{'value'" not in csv_text


def test_delete_input_after_success_true_deletes_input(tmp_path):
    config = _config(tmp_path)

    extract_pose_keypoints(config, _frames(1), _estimator)

    assert not Path(config.input_video_path).exists()


def test_delete_input_after_success_false_keeps_input(tmp_path):
    config = _config(tmp_path, delete=False)

    extract_pose_keypoints(config, _frames(1), _estimator)

    assert Path(config.input_video_path).exists()


def test_extraction_failure_does_not_delete_input_or_replace_output(tmp_path):
    config = _config(tmp_path)
    output_path = Path(config.output_csv_path)
    output_path.write_text("previous numeric output\n", encoding="utf-8")

    def failing_estimator(_frame):
        raise RuntimeError("synthetic estimator failure")

    with pytest.raises(RuntimeError, match="synthetic estimator failure"):
        extract_pose_keypoints(config, _frames(1), failing_estimator)

    assert Path(config.input_video_path).exists()
    assert output_path.read_text(encoding="utf-8") == "previous numeric output\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_max_frames_limits_extraction(tmp_path):
    config = _config(tmp_path, delete=False, max_frames=2)

    rows = extract_pose_keypoints(config, _frames(5), _estimator)

    assert [row.frame_index for row in rows] == [0, 1]


def test_extraction_creates_no_media_files_in_repository(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    forbidden_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}
    before = {
        path.relative_to(repository_root)
        for path in repository_root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    }
    config = _config(tmp_path, delete=False)

    extract_pose_keypoints(config, _frames(1), _estimator)

    after = {
        path.relative_to(repository_root)
        for path in repository_root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    }
    assert after == before


def test_mediapipe_tasks_fallback_when_solutions_is_unavailable(
    tmp_path, monkeypatch
):
    task_path = tmp_path / "pose.task"
    task_path.write_bytes(b"temporary task placeholder")

    landmark = SimpleNamespace(x=0.1, y=0.2, z=-0.3, visibility=0.8)

    class Detector:
        def detect(self, _image):
            return SimpleNamespace(pose_landmarks=[[landmark]])

        def close(self):
            return None

    fake_mp = SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: kwargs,
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE="image"),
                PoseLandmarkerOptions=lambda **kwargs: kwargs,
                PoseLandmarker=SimpleNamespace(
                    create_from_options=lambda _options: Detector()
                ),
            ),
        ),
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setenv("MEDIAPIPE_POSE_LANDMARKER_TASK", str(task_path))

    estimator = _MediaPipePoseEstimator()
    rows = estimator(np.zeros((2, 2, 3), dtype=np.uint8))

    assert estimator._backend == "tasks"
    assert rows == [PoseLandmark(0, 0.1, 0.2, -0.3, 0.8)]
    estimator.close()
