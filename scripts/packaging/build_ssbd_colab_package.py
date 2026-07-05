#!/usr/bin/env python3
"""Build a safe Colab upload package for the SSBD+ benchmark workflow."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Sequence


PACKAGE_ROOT_NAME = "ssbd_colab_package"
REPO_PACKAGE_NAME = "ssbd-stimming-recognition"

BLOCKED_SUFFIXES = {
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".gif",
    ".gz",
    ".h5",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".keras",
    ".mkv",
    ".mov",
    ".mp4",
    ".m4v",
    ".onnx",
    ".pkl",
    ".pb",
    ".png",
    ".pt",
    ".pth",
    ".task",
    ".tar",
    ".tflite",
    ".tif",
    ".tiff",
    ".webm",
    ".xz",
    ".zip",
    ".zst",
}

BLOCKED_PATH_DIRS = {
    "artifacts",
    "cache",
    "caches",
    "data",
    "frames",
    "images",
    "output",
    "outputs",
    "raw",
    "runs",
    "temp",
    "tmp",
    "videos",
}

BLOCKED_DIR_NAMES = {
    ".codebase-memory",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Colab upload package with the repository and metadata in the "
            "expected ssbd_colab_package/ layout. This command excludes raw media, "
            "model binaries, caches, outputs, artifacts, and git metadata."
        )
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    return parser


def build_ssbd_colab_package(
    repo_root: Path | str,
    metadata_dir: Path | str,
    output_zip: Path | str,
) -> tuple[str, ...]:
    repo_root = Path(repo_root).expanduser().resolve()
    metadata_dir = Path(metadata_dir).expanduser().resolve()
    destination = Path(output_zip).expanduser().resolve()

    if not repo_root.is_dir():
        raise NotADirectoryError(f"repo root does not exist: {repo_root}")
    if not metadata_dir.is_dir():
        raise NotADirectoryError(f"metadata dir does not exist: {metadata_dir}")
    if _path_within(destination, repo_root) or _path_within(destination, metadata_dir):
        raise ValueError("output zip must be outside the packaged source trees")

    destination.parent.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        members.extend(
            _write_tree(
                archive,
                repo_root,
                f"{PACKAGE_ROOT_NAME}/{REPO_PACKAGE_NAME}",
            )
        )
        members.extend(
            _write_tree(
                archive,
                metadata_dir,
                f"{PACKAGE_ROOT_NAME}/metadata",
            )
        )

    if not members:
        raise ValueError("no files were packaged")
    return tuple(members)


def _write_tree(
    archive: zipfile.ZipFile,
    source_root: Path,
    archive_prefix: str,
) -> list[str]:
    members: list[str] = []
    for path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        if _should_skip(path, source_root):
            continue
        member = f"{archive_prefix}/{path.relative_to(source_root).as_posix()}"
        archive.write(path, member)
        members.append(member)
    return members


def _should_skip(path: Path, source_root: Path) -> bool:
    if path.is_symlink():
        raise ValueError(f"symbolic links are not allowed in the package: {path}")

    relative = path.relative_to(source_root)
    if any(part in BLOCKED_PATH_DIRS or part in BLOCKED_DIR_NAMES for part in relative.parts):
        return True
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return True
    return False


def _path_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        members = build_ssbd_colab_package(args.repo_root, args.metadata_dir, args.output_zip)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(members)} packaged file(s) to {args.output_zip.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
