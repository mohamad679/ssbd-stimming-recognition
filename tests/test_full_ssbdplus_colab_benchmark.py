import csv
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "benchmark"
    / "run_full_ssbdplus_colab_benchmark.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("full_ssbdplus_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
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
        writer.writerows(rows)


def _segment(
    video_id: str,
    start: int,
    end: int,
    category: str,
) -> dict[str, str]:
    return {
        "xml_file_name": f"{video_id}.xml",
        "youtube_video_url": f"https://www.youtube.com/watch?v={video_id}",
        "action_start_time": str(start),
        "action_end_time": str(end),
        "action_category": category,
    }


def test_metadata_validation_checks_schema_counts_and_times(tmp_path):
    runner = _load_runner()
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, [_segment("action_1", 0, 3, "armflapping")])

    bundle = runner.validate_metadata(
        metadata, expected_segment_count=1, expected_video_count=1
    )

    assert len(bundle.segments) == 1
    with pytest.raises(ValueError, match="expected 2 segment rows"):
        runner.validate_metadata(metadata, expected_segment_count=2)

    _write_metadata(metadata, [_segment("action_1", 4, 3, "armflapping")])
    with pytest.raises(ValueError, match="0 <= start < end"):
        runner.validate_metadata(metadata)


def test_multi_category_video_preserves_each_segment_label(tmp_path):
    runner = _load_runner()
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            _segment("action_4", 0, 4, "armflapping"),
            _segment("action_4", 5, 11, "spinning"),
        ],
    )

    bundle = runner.validate_metadata(metadata)

    assert [segment.action_category for segment in bundle.segments] == [
        "armflapping",
        "spinning",
    ]
    assert bundle.videos[0].action_categories == ("armflapping", "spinning")
    assert bundle.videos[0].primary_action_category == "spinning"

    paths = runner.prepare_run_paths(tmp_path / "work")
    runner._write_metadata_artifacts(paths, bundle)
    manifest = json.loads(
        (paths.manifests / "metadata_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["videos"][0]["primary_action_category_usage"] == "reporting_only"
    with (paths.features / "segment_labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    assert len(labels) == 2
    assert labels[0]["action_category_code"] != labels[1]["action_category_code"]


def test_resume_skips_only_valid_nonempty_csv(tmp_path):
    runner = _load_runner()
    output = tmp_path / "keypoints.csv"
    output.write_text(
        "video_id,frame_index,timestamp_s,landmark_index,x,y,z,confidence\n"
        "action_1,0,0.0,0,0.5,0.5,0.0,0.9\n",
        encoding="utf-8",
    )

    assert runner.should_skip_completed_output(
        output,
        resume=True,
        required_columns=runner.KEYPOINT_COLUMNS,
    )
    assert not runner.should_skip_completed_output(
        output,
        resume=False,
        required_columns=runner.KEYPOINT_COLUMNS,
    )
    output.write_text(
        "video_id,frame_index,timestamp_s,landmark_index,x,y,z,confidence\n",
        encoding="utf-8",
    )
    assert not runner.should_skip_completed_output(
        output,
        resume=True,
        required_columns=runner.KEYPOINT_COLUMNS,
    )
    output.write_text(
        "video_id,frame_index,timestamp_s,landmark_index,x,y,z,confidence\n"
        "action_1,not-an-index,0.0,0,0.5,0.5,0.0,0.9\n",
        encoding="utf-8",
    )
    assert not runner.should_skip_completed_output(
        output,
        resume=True,
        required_columns=runner.KEYPOINT_COLUMNS,
    )


def test_final_zip_contains_only_safe_allowlisted_artifacts(tmp_path):
    runner = _load_runner()
    artifacts = tmp_path / "artifacts"
    (artifacts / "keypoints").mkdir(parents=True)
    (artifacts / "reports").mkdir()
    (artifacts / "keypoints" / "one.csv").write_text("x\n1\n", encoding="utf-8")
    (artifacts / "reports" / "summary.json").write_text("{}\n", encoding="utf-8")
    (artifacts / "reports" / "metrics.txt").write_text("aggregate\n", encoding="utf-8")
    (artifacts / "reports" / "pose.svg").write_text("<svg/>\n", encoding="utf-8")
    output_zip = tmp_path / "safe.zip"

    members = runner.create_safe_zip(artifacts, output_zip)

    assert set(members) == {
        "keypoints/one.csv",
        "reports/summary.json",
        "reports/metrics.txt",
        "reports/pose.svg",
    }
    with zipfile.ZipFile(output_zip) as archive:
        assert set(archive.namelist()) == set(members)
        assert all(
            Path(name).suffix.lower() in runner.SAFE_ARCHIVE_SUFFIXES
            for name in archive.namelist()
        )


def test_exit_policy_allows_reported_video_failures_after_successful_pipeline():
    runner = _load_runner()

    assert runner.benchmark_exit_code(
        feature_table_count=1,
        evaluation_completed=True,
        safe_zip_created=True,
    ) == 0


@pytest.mark.parametrize(
    ("feature_table_count", "evaluation_completed", "safe_zip_created"),
    [
        (0, True, True),
        (1, False, True),
        (1, True, False),
    ],
)
def test_exit_policy_rejects_fatal_pipeline_failures(
    feature_table_count, evaluation_completed, safe_zip_created
):
    runner = _load_runner()

    assert runner.benchmark_exit_code(
        feature_table_count=feature_table_count,
        evaluation_completed=evaluation_completed,
        safe_zip_created=safe_zip_created,
    ) != 0


@pytest.mark.parametrize(
    "unsafe_name",
    ["raw.mp4", "frame.jpg", "image.png", "pose.task", "model.pkl", "model.pt"],
)
def test_final_zip_rejects_media_image_task_and_model_artifacts(tmp_path, unsafe_name):
    runner = _load_runner()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "safe.csv").write_text("x\n1\n", encoding="utf-8")
    (artifacts / unsafe_name).write_bytes(b"unsafe")

    with pytest.raises(ValueError, match="forbidden"):
        runner.create_safe_zip(artifacts, tmp_path / "results.zip")
    assert not (tmp_path / "results.zip").exists()
