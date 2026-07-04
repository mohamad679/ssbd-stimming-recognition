#!/usr/bin/env python3
"""Validate existing SVG files as well-formed XML without generating figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.evaluation import validate_svg_files  # noqa: E402


NON_SVG_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate existing SVG files as well-formed XML only. "
            "This command does not generate figures, download videos, or run MediaPipe."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="explicit .svg file paths")
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="recursively validate all .svg files under a directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        svg_paths = _collect_svg_paths(args.paths, args.directory)
        validated_paths = validate_svg_files(svg_paths)
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Validated {len(validated_paths)} SVG file(s).")
    return 0


def _collect_svg_paths(paths: Sequence[Path], directory: Path | None) -> list[Path]:
    if not paths and directory is None:
        raise ValueError("provide at least one .svg path or --directory")

    svg_paths = []
    for path in paths:
        _validate_explicit_svg_path(path)
        svg_paths.append(path)

    if directory is not None:
        if not directory.exists():
            raise FileNotFoundError(f"SVG directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"SVG directory must be a directory: {directory}")

        directory_paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".svg"
        )
        if not directory_paths:
            raise ValueError(f"no .svg files found under directory: {directory}")
        svg_paths.extend(directory_paths)

    return svg_paths


def _validate_explicit_svg_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return
    if suffix in NON_SVG_IMAGE_EXTENSIONS:
        raise ValueError(f"non-SVG image formats are not supported: {path}")
    raise ValueError(f"explicit paths must point to .svg files: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
