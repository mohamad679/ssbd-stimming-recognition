from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from ssbd_behavior.evaluation import validate_svg_file
from ssbd_behavior.interpretability import (
    SkeletonPoint2D,
    render_sequence_summary_svg,
    render_skeleton_svg,
)
from ssbd_behavior.pose import PoseKeypoint, write_keypoints_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "render_skeleton_svg.py"
FIGURE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


def test_render_skeleton_svg_returns_well_formed_svg() -> None:
    svg_text = render_skeleton_svg(_synthetic_pose())
    root = ET.fromstring(svg_text)

    assert _local_name(root.tag) == "svg"
    assert root.findall(".//{http://www.w3.org/2000/svg}line")
    assert root.findall(".//{http://www.w3.org/2000/svg}circle")


def test_generated_svg_passes_validation_when_written(tmp_path) -> None:
    svg_path = tmp_path / "skeleton.svg"
    svg_path.write_text(render_sequence_summary_svg(_synthetic_sequence()), encoding="utf-8")

    assert validate_svg_file(svg_path) == svg_path


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one keypoint"):
        render_skeleton_svg([])


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_coordinates_are_rejected(coordinate) -> None:
    with pytest.raises(ValueError, match="finite numeric coordinate"):
        SkeletonPoint2D(0, coordinate, 0.2, 0.9)


def test_low_confidence_and_missing_points_are_skipped_without_crashing() -> None:
    svg_text = render_skeleton_svg(
        [
            SkeletonPoint2D(0, 0.50, 0.10, 0.95),
            SkeletonPoint2D(11, 0.30, 0.30, 0.95),
            SkeletonPoint2D(12, 0.70, 0.30, 0.20),
            SkeletonPoint2D(13, None, None, 0.0),
            SkeletonPoint2D(15, 0.20, 0.60, 0.95),
        ]
    )

    root = ET.fromstring(svg_text)
    circles = root.findall(".//{http://www.w3.org/2000/svg}circle")
    assert _local_name(root.tag) == "svg"
    assert 1 <= len(circles) < 5


def test_cli_dry_run_writes_nothing(tmp_path) -> None:
    csv_path = tmp_path / "synthetic-keypoints.csv"
    output_path = tmp_path / "dry-run.svg"
    write_keypoints_csv(csv_path, _synthetic_csv_rows())

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(csv_path), "--output", str(output_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Dry run:" in result.stdout
    assert not output_path.exists()


def test_cli_execute_writes_svg_only_to_output_path(tmp_path) -> None:
    csv_path = tmp_path / "synthetic-keypoints.csv"
    output_path = tmp_path / "rendered.svg"
    write_keypoints_csv(csv_path, _synthetic_csv_rows())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(csv_path),
            "--output",
            str(output_path),
            "--execute",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()
    assert validate_svg_file(output_path) == output_path
    assert result.stderr == ""


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"])
def test_cli_rejects_image_and_video_extensions(tmp_path, suffix) -> None:
    input_path = tmp_path / f"suspicious{suffix}"
    output_path = tmp_path / "output.svg"
    input_path.write_text("not numeric keypoints", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_path),
            "--output",
            str(output_path),
            "--execute",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "media/image inputs are not supported" in result.stderr
    assert not output_path.exists()


def test_cli_does_not_create_generated_svg_or_images_inside_repo(tmp_path) -> None:
    csv_path = tmp_path / "synthetic-keypoints.csv"
    output_path = tmp_path / "external-output.svg"
    write_keypoints_csv(csv_path, _synthetic_csv_rows())
    before_paths = _repo_figure_paths()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(csv_path),
            "--output",
            str(output_path),
            "--execute",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    after_paths = _repo_figure_paths()

    assert result.returncode == 0
    assert output_path.exists()
    assert before_paths == after_paths


def _synthetic_pose() -> list[SkeletonPoint2D]:
    return [
        SkeletonPoint2D(0, 0.50, 0.08, 0.95),
        SkeletonPoint2D(11, 0.35, 0.25, 0.95),
        SkeletonPoint2D(12, 0.65, 0.25, 0.95),
        SkeletonPoint2D(13, 0.25, 0.45, 0.95),
        SkeletonPoint2D(14, 0.75, 0.45, 0.95),
        SkeletonPoint2D(15, 0.18, 0.65, 0.95),
        SkeletonPoint2D(16, 0.82, 0.65, 0.95),
        SkeletonPoint2D(23, 0.40, 0.58, 0.95),
        SkeletonPoint2D(24, 0.60, 0.58, 0.95),
        SkeletonPoint2D(25, 0.38, 0.82, 0.95),
        SkeletonPoint2D(26, 0.62, 0.82, 0.95),
        SkeletonPoint2D(27, 0.35, 1.00, 0.95),
        SkeletonPoint2D(28, 0.65, 1.00, 0.95),
    ]


def _synthetic_sequence() -> list[list[SkeletonPoint2D]]:
    first_pose = _synthetic_pose()
    return [
        first_pose,
        [
            SkeletonPoint2D(
                point.landmark_index,
                None if point.x is None else point.x + 0.03,
                point.y,
                point.confidence,
            )
            for point in first_pose
        ],
        [
            SkeletonPoint2D(
                point.landmark_index,
                point.x,
                None if point.y is None else point.y + 0.02,
                point.confidence,
            )
            for point in first_pose
        ],
    ]


def _synthetic_csv_rows() -> list[PoseKeypoint]:
    return [
        PoseKeypoint("synthetic-1", 0, 0.0, 0, 0.50, 0.08, confidence=0.95),
        PoseKeypoint("synthetic-1", 0, 0.0, 11, 0.35, 0.25, confidence=0.95),
        PoseKeypoint("synthetic-1", 0, 0.0, 12, 0.65, 0.25, confidence=0.95),
        PoseKeypoint("synthetic-1", 0, 0.0, 15, 0.18, 0.65, confidence=0.95),
        PoseKeypoint("synthetic-1", 0, 0.0, 16, 0.82, 0.65, confidence=0.95),
        PoseKeypoint("synthetic-1", 1, 1 / 30, 0, 0.52, 0.10, confidence=0.95),
        PoseKeypoint("synthetic-1", 1, 1 / 30, 11, 0.36, 0.27, confidence=0.95),
        PoseKeypoint("synthetic-1", 1, 1 / 30, 12, 0.64, 0.27, confidence=0.95),
        PoseKeypoint("synthetic-1", 1, 1 / 30, 15, 0.20, 0.66, confidence=0.95),
        PoseKeypoint("synthetic-1", 1, 1 / 30, 16, 0.80, 0.66, confidence=0.95),
    ]


def _repo_figure_paths() -> set[Path]:
    return {
        path.relative_to(REPOSITORY_ROOT)
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FIGURE_EXTENSIONS
    }


def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", maxsplit=1)[1]
    return tag
