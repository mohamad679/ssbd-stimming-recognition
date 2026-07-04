"""Validation helpers for SVG files stored as XML documents."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_svg_file(path: str | Path) -> Path:
    """Validate that a file is well-formed XML with an ``svg`` root element."""

    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"SVG file does not exist: {candidate}")
    if candidate.is_dir():
        raise IsADirectoryError(f"SVG path must be a file, not a directory: {candidate}")

    try:
        root = ET.parse(candidate).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"malformed SVG XML in {candidate}: {exc}") from exc

    root_name = _local_name(root.tag)
    if root_name != "svg":
        raise ValueError(
            f"expected root element <svg> in {candidate}, found <{root_name}>"
        )

    return candidate


def validate_svg_files(paths: Iterable[str | Path]) -> list[Path]:
    """Validate multiple SVG files and return the validated paths in input order."""

    validated_paths = [validate_svg_file(path) for path in paths]
    if not validated_paths:
        raise ValueError("at least one SVG file path is required")
    return validated_paths


def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", maxsplit=1)[1]
    return tag
