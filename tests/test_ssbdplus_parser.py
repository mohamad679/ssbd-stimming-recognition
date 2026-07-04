import csv

import pytest

from ssbd_behavior.acquisition import (
    load_ssbdplus_csv,
    parse_ssbdplus_xml,
    parse_time_range,
    summarize_segments,
)


def test_parse_time_range():
    assert parse_time_range("3:5") == (3, 5)
    assert parse_time_range(" 10 : 14 ") == (10, 14)


@pytest.mark.parametrize("value", ["", "3", "3:3", "5:3", "-1:3", "1.5:3"])
def test_parse_time_range_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_time_range(value)


def test_load_ssbdplus_csv(tmp_path):
    csv_path = tmp_path / "segments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "xml_file_name",
                "youtube_video_url",
                "action_start_time",
                "action_end_time",
                "action_category",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "xml_file_name": "action_1.xml",
                "youtube_video_url": "https://example.invalid/video-1",
                "action_start_time": "3",
                "action_end_time": "5",
                "action_category": "armflapping",
            }
        )

    segments = load_ssbdplus_csv(csv_path)

    assert len(segments) == 1
    assert segments[0].video_id == "action_1"
    assert segments[0].start_time == 3
    assert segments[0].end_time == 5
    assert segments[0].category == "armflapping"
    assert segments[0].annotation_file == "action_1.xml"


def test_load_ssbdplus_csv_requires_observed_columns(tmp_path):
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text("xml_file_name,action_category\naction_1.xml,armflapping\n")

    with pytest.raises(ValueError, match="missing required CSV columns"):
        load_ssbdplus_csv(csv_path)


def test_parse_ssbdplus_xml(tmp_path):
    xml_path = tmp_path / "action_1.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<video id="video-1">
  <url>https://example.invalid/video-1</url>
  <behaviours count="2">
    <behaviour id="1"><time>3:5</time><category>armflapping</category></behaviour>
    <behaviour id="2"><time>8:12</time><category>armflapping</category></behaviour>
  </behaviours>
</video>
""",
        encoding="utf-8",
    )

    segments = parse_ssbdplus_xml(xml_path)

    assert [(segment.start_time, segment.end_time) for segment in segments] == [
        (3, 5),
        (8, 12),
    ]
    assert {segment.video_id for segment in segments} == {"video-1"}
    assert [segment.behaviour_id for segment in segments] == ["1", "2"]
    assert {segment.category for segment in segments} == {"armflapping"}


def test_summarize_segments(tmp_path):
    xml_path = tmp_path / "action_1.xml"
    xml_path.write_text(
        """<video id="video-1">
  <url>https://example.invalid/video-1</url>
  <behaviours>
    <behaviour id="1"><time>1:2</time><category>armflapping</category></behaviour>
    <behaviour id="2"><time>3:4</time><category>spinning</category></behaviour>
    <behaviour id="3"><time>5:6</time><category>armflapping</category></behaviour>
  </behaviours>
</video>""",
        encoding="utf-8",
    )

    summary = summarize_segments(parse_ssbdplus_xml(xml_path))

    assert summary == {
        "unique_videos": 1,
        "total_segments": 3,
        "category_counts": {"armflapping": 2, "spinning": 1},
    }
