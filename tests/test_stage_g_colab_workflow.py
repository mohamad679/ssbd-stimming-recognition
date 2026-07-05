import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from ssbd_behavior.features import LEFT_WRIST, NOSE, RIGHT_WRIST
from ssbd_behavior.pose import PoseKeypoint, write_keypoints_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "benchmark"
    / "run_stage_g_d_ms_stf_colab.py"
)


def _load_workflow():
    spec = importlib.util.spec_from_file_location("stage_g_workflow", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_keypoints(video_id: str, *, frames: int = 40) -> list[PoseKeypoint]:
    points = []
    for frame in range(frames):
        timestamp = frame / 10.0
        phase = 2.0 * np.pi * timestamp
        for landmark, x, y in (
            (NOSE, 0.5, 0.4 + 0.02 * np.sin(phase)),
            (LEFT_WRIST, 0.4 + 0.08 * np.sin(phase), 0.6),
            (RIGHT_WRIST, 0.6 - 0.08 * np.sin(phase), 0.6),
        ):
            points.append(
                PoseKeypoint(
                    video_id=video_id,
                    frame_index=frame,
                    timestamp_s=timestamp,
                    landmark_index=landmark,
                    x=x,
                    y=y,
                    confidence=0.9,
                )
            )
    return points


def _write_features(path: Path, video_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "video_id",
                "window_start_s",
                "window_end_s",
                "label",
                "baseline_motion",
                "baseline_periodicity",
            ),
        )
        writer.writeheader()
        for group_index, video_id in enumerate(video_ids):
            for sample_index, label in enumerate((0, 1, 0, 1)):
                writer.writerow(
                    {
                        "video_id": video_id,
                        "window_start_s": sample_index,
                        "window_end_s": sample_index + 1,
                        "label": label,
                        "baseline_motion": label + group_index * 0.01,
                        "baseline_periodicity": 0.8 * label + sample_index * 0.01,
                    }
                )


def test_multiscale_augmentation_uses_numeric_keypoints_and_window_ends(
    tmp_path,
):
    workflow = _load_workflow()
    keypoints = tmp_path / "keypoints.csv"
    features = tmp_path / "features.csv"
    output = tmp_path / "features_with_ms.csv"
    write_keypoints_csv(keypoints, _synthetic_keypoints("synthetic-0"))
    _write_features(features, ["synthetic-0"])

    augmented = workflow.augment_multiscale_feature_csv(
        features,
        output,
        keypoints=keypoints,
    )

    assert augmented is True
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ms_columns = [name for name in rows[0] if name.startswith("ms_")]
    assert len(ms_columns) == 42
    assert all(np.isfinite(float(row[name])) for row in rows for name in ms_columns)
    assert rows[0]["ms_1s__motion_energy"] != rows[-1]["ms_1s__motion_energy"]

    reused_output = tmp_path / "reused_features_with_ms.csv"
    reused = workflow.augment_multiscale_feature_csv(
        output,
        reused_output,
        keypoints=None,
    )
    assert reused is False
    assert reused_output.read_text(encoding="utf-8") == output.read_text(
        encoding="utf-8"
    )


def test_stage_g_workflow_runs_smoke_mode_without_raw_videos(tmp_path):
    keypoint_dir = tmp_path / "numeric-keypoints"
    keypoint_dir.mkdir()
    video_ids = [f"synthetic-{index}" for index in range(6)]
    for video_id in video_ids:
        write_keypoints_csv(
            keypoint_dir / f"{video_id}.csv", _synthetic_keypoints(video_id)
        )
    features = tmp_path / "features.csv"
    _write_features(features, video_ids)
    output_dir = tmp_path / "stage_g_results"
    output_zip = tmp_path / "stage_g_results.zip"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--feature-csv",
            str(features),
            "--keypoints",
            str(keypoint_dir),
            "--output-dir",
            str(output_dir),
            "--group-splits",
            "3",
            "--inner-splits",
            "2",
            "--n-estimators",
            "4",
            "--smoke",
            "--create-zip",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Permutation count: 2" in completed.stdout
    assert {
        "aggregate_metrics.csv",
        "features_with_ms.csv",
        "fold_metrics.csv",
        "report.json",
    } == {path.name for path in output_dir.iterdir()}
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert {row["protocol"] for row in report["aggregates"]} == {
        "group_kfold",
        "loso",
    }
    assert report["permutation"]["n_permutations"] == 2
    assert output_zip.is_file()
    assert not any(path.suffix.lower() in {".mp4", ".png", ".task", ".pkl"} for path in tmp_path.rglob("*"))


def test_safe_stage_g_packaging_excludes_forbidden_file_patterns(tmp_path):
    workflow = _load_workflow()
    results = tmp_path / "results"
    results.mkdir()
    (results / "aggregate_metrics.csv").write_text("metric,value\nauroc,0.5\n")
    (results / "fold_metrics.csv").write_text("fold,value\n0,0.5\n")
    (results / "report.json").write_text("{}\n")
    unsafe_names = (
        "raw_video.mp4",
        "frame.jpg",
        "image.png",
        "pose.task",
        "model.pkl",
        "weights.pt",
    )
    for name in unsafe_names:
        (results / name).write_bytes(b"unsafe")
    (results / "frames").mkdir()
    (results / "frames" / "numeric-looking.csv").write_text("x\n1\n")
    output_zip = tmp_path / "safe.zip"

    members = workflow.create_safe_results_zip(results, output_zip)

    assert set(members) == {
        "aggregate_metrics.csv",
        "fold_metrics.csv",
        "report.json",
    }
    with zipfile.ZipFile(output_zip) as archive:
        assert set(archive.namelist()) == set(members)
        assert not any(name in member for name in unsafe_names for member in members)
