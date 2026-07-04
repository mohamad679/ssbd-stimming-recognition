import csv
import math
from pathlib import Path
import subprocess
import sys

from ssbd_behavior.acquisition import SSBDPlusSegment
from ssbd_behavior.features import (
    LEFT_WRIST,
    NOSE,
    RIGHT_WRIST,
    WindowSpec,
    build_feature_rows_for_video,
    read_feature_table_csv,
    write_feature_table_csv,
)
from ssbd_behavior.pose import PoseKeypoint, write_keypoints_csv


def _synthetic_keypoints(
    video_id: str, *, frame_count: int = 40, sample_rate_hz: float = 10.0
) -> list[PoseKeypoint]:
    rows = []
    for frame in range(frame_count):
        timestamp_s = frame / sample_rate_hz
        signal = math.sin(2.0 * math.pi * timestamp_s)
        rows.extend(
            [
                PoseKeypoint(
                    video_id, frame, timestamp_s, LEFT_WRIST, signal, 0.0, 0.0, 1.0
                ),
                PoseKeypoint(
                    video_id,
                    frame,
                    timestamp_s,
                    RIGHT_WRIST,
                    signal + 1.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                PoseKeypoint(
                    video_id, frame, timestamp_s, NOSE, 0.0, signal, 0.0, 1.0
                ),
            ]
        )
    return rows


def _segment(video_id: str, start: int, end: int) -> SSBDPlusSegment:
    return SSBDPlusSegment(
        video_id=video_id,
        url="https://example.invalid/synthetic-only",
        start_time=start,
        end_time=end,
        category="armflapping",
    )


def test_builds_one_labeled_feature_row_per_window_for_one_video() -> None:
    rows = build_feature_rows_for_video(
        video_id="video-1",
        keypoint_rows=_synthetic_keypoints("video-1"),
        annotations=[_segment("video-1", 0, 2)],
        spec=WindowSpec(window_size_s=2.0, stride_s=1.0),
        sample_rate_hz=10.0,
        minimum_overlap_fraction=0.5,
    )

    assert len(rows) == 3
    assert [(row.window_start_s, row.window_end_s) for row in rows] == [
        (0.0, 2.0),
        (1.0, 3.0),
        (2.0, 4.0),
    ]
    assert [row.label for row in rows] == [1, 1, 0]
    assert all(row.video_id == "video-1" for row in rows)
    assert all(all(type(value) is float for value in row.features.values()) for row in rows)


def test_keypoints_from_another_video_are_ignored() -> None:
    target_rows = _synthetic_keypoints("video-1", frame_count=20)
    other_rows = _synthetic_keypoints("video-2", frame_count=100)

    rows = build_feature_rows_for_video(
        video_id="video-1",
        keypoint_rows=target_rows + other_rows,
        annotations=[_segment("video-2", 0, 10)],
        spec=WindowSpec(window_size_s=2.0, stride_s=1.0),
        sample_rate_hz=10.0,
    )

    assert len(rows) == 1
    assert rows[0].video_id == "video-1"
    assert rows[0].label == 0


def test_feature_table_csv_roundtrip_is_numeric_and_privacy_safe(tmp_path) -> None:
    rows = build_feature_rows_for_video(
        video_id="video-1",
        keypoint_rows=_synthetic_keypoints("video-1"),
        annotations=[_segment("video-1", 0, 2)],
        spec=WindowSpec(window_size_s=2.0, stride_s=2.0),
        sample_rate_hz=10.0,
    )
    output_path = tmp_path / "features.csv"

    write_feature_table_csv(output_path, rows)

    assert read_feature_table_csv(output_path) == rows
    csv_text = output_path.read_text(encoding="utf-8")
    lowered = csv_text.lower()
    for forbidden in ("video_path", "raw_frame", "image", "source_url", "https://"):
        assert forbidden not in lowered

    with output_path.open(encoding="utf-8", newline="") as stream:
        persisted_rows = list(csv.DictReader(stream))
    numeric_columns = set(persisted_rows[0]) - {"video_id"}
    for persisted_row in persisted_rows:
        assert persisted_row["video_id"] == "video-1"
        assert all(math.isfinite(float(persisted_row[name])) for name in numeric_columns)


def test_cli_dry_run_and_local_synthetic_csv_end_to_end(tmp_path) -> None:
    keypoint_path = tmp_path / "keypoints.csv"
    annotation_path = tmp_path / "annotations.csv"
    output_path = tmp_path / "features.csv"
    write_keypoints_csv(
        keypoint_path, _synthetic_keypoints("video-1", frame_count=20)
    )
    with annotation_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "xml_file_name",
                "youtube_video_url",
                "action_start_time",
                "action_end_time",
                "action_category",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "xml_file_name": "video-1.xml",
                "youtube_video_url": "https://example.invalid/synthetic-only",
                "action_start_time": "0",
                "action_end_time": "2",
                "action_category": "armflapping",
            }
        )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "features"
        / "build_feature_table.py"
    )
    command = [
        sys.executable,
        str(script),
        str(keypoint_path),
        str(output_path),
        "--annotations-csv",
        str(annotation_path),
        "--window-size-s",
        "2",
        "--stride-s",
        "1",
        "--sample-rate-hz",
        "10",
    ]

    dry_run = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Dry run" in dry_run.stderr
    assert not output_path.exists()

    subprocess.run(command + ["--execute"], check=True, capture_output=True, text=True)
    output_rows = read_feature_table_csv(output_path)
    assert len(output_rows) == 1
    assert output_rows[0].label == 1
