import csv
from io import StringIO

import pytest

from ssbd_behavior.acquisition import (
    AccessReportRow,
    DatasetSource,
    ManifestEntry,
    render_access_report_csv,
    write_access_report_csv,
)


def test_manifest_entry_creates_unattempted_report_row():
    entry = ManifestEntry(
        video_id="plus-001",
        source=DatasetSource.SSBD_PLUS,
        reference="manifest.xml#plus-001",
        annotation_status="present",
    )

    row = entry.to_access_report_row()

    assert row.attempted_video_id == "plus-001"
    assert row.download_status == "not_attempted"
    assert row.usable_segment_count == 0


def test_access_report_renders_stable_csv_schema():
    row = AccessReportRow(
        attempted_video_id="original-001",
        source=DatasetSource.ORIGINAL_SSBD,
        url_or_manifest_reference="url-list.pdf#1",
        download_status="unavailable",
        failure_reason="removed",
        annotation_status="present",
        notes="link-rot finding",
    )

    records = list(csv.DictReader(StringIO(render_access_report_csv([row]))))

    assert records == [
        {
            "attempted_video_id": "original-001",
            "source": "original SSBD",
            "url_or_manifest_reference": "url-list.pdf#1",
            "download_status": "unavailable",
            "failure_reason": "removed",
            "annotation_status": "present",
            "usable_segment_count": "0",
            "notes": "link-rot finding",
        }
    ]


def test_access_report_rejects_negative_segment_count():
    with pytest.raises(ValueError, match="must not be negative"):
        AccessReportRow(
            attempted_video_id="plus-002",
            source=DatasetSource.SSBD_PLUS,
            url_or_manifest_reference="manifest.xml#plus-002",
            download_status="not_attempted",
            usable_segment_count=-1,
        )


def test_access_report_can_be_written_to_csv(tmp_path):
    output_path = tmp_path / "reports" / "access.csv"
    row = ManifestEntry(
        video_id="plus-003",
        source=DatasetSource.SSBD_PLUS,
        reference="manifest.xml#plus-003",
    ).to_access_report_row()

    written_path = write_access_report_csv([row], output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8").startswith("attempted_video_id,")
