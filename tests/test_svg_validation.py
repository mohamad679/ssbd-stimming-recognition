from pathlib import Path
import subprocess
import sys

import pytest

from ssbd_behavior.evaluation import validate_svg_file, validate_svg_files


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "validate_svgs.py"
FIGURE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


def test_valid_svg_passes(tmp_path) -> None:
    first_path = tmp_path / "first.svg"
    second_path = tmp_path / "second.svg"
    first_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10" /></svg>',
        encoding="utf-8",
    )
    second_path.write_text("<svg><g /></svg>", encoding="utf-8")

    assert validate_svg_file(first_path) == first_path
    assert validate_svg_files([first_path, second_path]) == [first_path, second_path]


def test_malformed_svg_fails(tmp_path) -> None:
    svg_path = tmp_path / "broken.svg"
    svg_path.write_text("<svg><g></svg>", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed SVG XML"):
        validate_svg_file(svg_path)


def test_xml_with_wrong_root_fails(tmp_path) -> None:
    svg_path = tmp_path / "wrong-root.svg"
    svg_path.write_text("<html><body /></html>", encoding="utf-8")

    with pytest.raises(ValueError, match=r"expected root element <svg>"):
        validate_svg_file(svg_path)


def test_cli_validates_explicit_file(tmp_path) -> None:
    svg_path = tmp_path / "figure.svg"
    svg_path.write_text("<svg><circle r='4' /></svg>", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(svg_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "Validated 1 SVG file(s)."
    assert result.stderr == ""


def test_cli_validates_directory_recursively(tmp_path) -> None:
    figures_dir = tmp_path / "figures"
    nested_dir = figures_dir / "nested"
    nested_dir.mkdir(parents=True)
    (figures_dir / "top.svg").write_text("<svg><rect /></svg>", encoding="utf-8")
    (nested_dir / "inner.svg").write_text("<svg><path /></svg>", encoding="utf-8")
    (nested_dir / "ignored.png").write_text("not an svg", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--directory", str(figures_dir)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "Validated 2 SVG file(s)."
    assert result.stderr == ""


def test_cli_fails_on_invalid_file(tmp_path) -> None:
    svg_path = tmp_path / "broken.svg"
    svg_path.write_text("<svg><g></svg>", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(svg_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "malformed SVG XML" in result.stderr


def test_cli_rejects_non_svg_image_file(tmp_path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_text("not an svg", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(image_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "non-SVG image formats are not supported" in result.stderr


def test_cli_does_not_create_figure_files_inside_repo(tmp_path) -> None:
    svg_path = tmp_path / "read-only.svg"
    svg_path.write_text("<svg><line /></svg>", encoding="utf-8")
    before_paths = _repo_figure_paths()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(svg_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    after_paths = _repo_figure_paths()

    assert result.returncode == 0
    assert before_paths == after_paths


def _repo_figure_paths() -> set[Path]:
    return {
        path.relative_to(REPOSITORY_ROOT)
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FIGURE_EXTENSIONS
    }
