"""Offline parsers for SSBD+ annotation metadata."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import defusedxml.ElementTree as ET


_CSV_COLUMNS = frozenset(
    {
        "xml_file_name",
        "youtube_video_url",
        "action_start_time",
        "action_end_time",
        "action_category",
    }
)
_TIME_RANGE_PATTERN = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


@dataclass(frozen=True)
class SSBDPlusSegment:
    """One annotated SSBD+ behaviour segment."""

    video_id: str
    url: str
    start_time: int | float
    end_time: int | float
    category: str
    annotation_file: str | None = None
    behaviour_id: str | None = None

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if not self.url.strip():
            raise ValueError("url must not be empty")
        if not self.category.strip():
            raise ValueError("category must not be empty")
        if self.start_time < 0 or self.start_time >= self.end_time:
            raise ValueError("segment times must satisfy 0 <= start_time < end_time")


def parse_time_range(value: str) -> tuple[int, int]:
    """Parse an SSBD+ ``start:end`` range expressed in integer seconds."""

    if not isinstance(value, str):
        raise ValueError("time range must be a string in start:end format")

    match = _TIME_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid time range {value!r}; expected start:end integers")

    start, end = (int(part) for part in match.groups())
    if start >= end:
        raise ValueError("time range must satisfy start < end")
    return start, end


def load_ssbdplus_csv(path: Path | str) -> list[SSBDPlusSegment]:
    """Load SSBD+ segment metadata from the observed CSV schema."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(_CSV_COLUMNS - columns)
        if missing:
            raise ValueError(f"missing required CSV columns: {', '.join(missing)}")

        segments: list[SSBDPlusSegment] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                start = _parse_second(row["action_start_time"])
                end = _parse_second(row["action_end_time"])
                if start >= end:
                    raise ValueError("start time must be less than end time")

                xml_file_name = _required_text(row["xml_file_name"], "xml_file_name")
                segments.append(
                    SSBDPlusSegment(
                        video_id=Path(xml_file_name).stem,
                        url=_required_text(
                            row["youtube_video_url"], "youtube_video_url"
                        ),
                        start_time=start,
                        end_time=end,
                        category=_required_text(
                            row["action_category"], "action_category"
                        ),
                        annotation_file=xml_file_name,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid SSBD+ CSV row {row_number}: {error}") from error

    return segments


def parse_ssbdplus_xml(path: Path | str) -> list[SSBDPlusSegment]:
    """Parse one local SSBD+ XML annotation file without network access."""

    xml_path = Path(path)
    root = ET.parse(xml_path).getroot()

    video_id = root.get("id") or _child_text(root, "id")
    if video_id is None:
        raise ValueError("SSBD+ XML video is missing its id")
    url = _child_text(root, "url")
    if url is None:
        raise ValueError("SSBD+ XML video is missing its url")

    behaviours = _child(root, "behaviours")
    if behaviours is None:
        raise ValueError("SSBD+ XML video is missing behaviours")

    segments: list[SSBDPlusSegment] = []
    for index, behaviour in enumerate(_children(behaviours, "behaviour"), start=1):
        time_value = _child_text(behaviour, "time")
        category = _child_text(behaviour, "category")
        if time_value is None or category is None:
            raise ValueError(f"SSBD+ XML behaviour {index} is missing time or category")
        start, end = parse_time_range(time_value)
        segments.append(
            SSBDPlusSegment(
                video_id=video_id.strip(),
                url=url.strip(),
                start_time=start,
                end_time=end,
                category=category.strip(),
                annotation_file=xml_path.name,
                behaviour_id=behaviour.get("id"),
            )
        )

    return segments


def summarize_segments(
    segments: Iterable[SSBDPlusSegment],
) -> dict[str, int | dict[str, int]]:
    """Summarize video, segment, and category counts."""

    segment_list = list(segments)
    return {
        "unique_videos": len({segment.video_id for segment in segment_list}),
        "total_segments": len(segment_list),
        "category_counts": dict(
            sorted(Counter(segment.category for segment in segment_list).items())
        ),
    }


def _parse_second(value: str | None) -> int | float:
    text = _required_text(value, "time")
    try:
        second = float(text)
    except ValueError as error:
        raise ValueError(f"invalid numeric second {value!r}") from error
    if not math.isfinite(second):
        raise ValueError(f"second must be finite, got {value!r}")
    if second < 0:
        raise ValueError(f"second must be non-negative, got {value!r}")
    return int(second) if second.is_integer() else second


def _required_text(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child) == name), None)


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child) == name]


def _child_text(element: ET.Element, name: str) -> str | None:
    child = _child(element, name)
    if child is None or child.text is None or not child.text.strip():
        return None
    return child.text.strip()
