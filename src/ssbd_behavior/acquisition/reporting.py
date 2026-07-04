"""CSV access-report representations and serialization helpers."""

from dataclasses import asdict, dataclass
import csv
from io import StringIO
from pathlib import Path
from typing import Iterable

from .manifest import DatasetSource


REPORT_COLUMNS = (
    "attempted_video_id",
    "source",
    "url_or_manifest_reference",
    "download_status",
    "failure_reason",
    "annotation_status",
    "usable_segment_count",
    "notes",
)


@dataclass(frozen=True, slots=True)
class AccessReportRow:
    """Observed or planned access status for one manifest entry."""

    attempted_video_id: str
    source: DatasetSource
    url_or_manifest_reference: str
    download_status: str
    failure_reason: str = ""
    annotation_status: str = "unknown"
    usable_segment_count: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.attempted_video_id.strip():
            raise ValueError("attempted_video_id must not be empty")
        if not self.url_or_manifest_reference.strip():
            raise ValueError("url_or_manifest_reference must not be empty")
        if not self.download_status.strip():
            raise ValueError("download_status must not be empty")
        if self.usable_segment_count < 0:
            raise ValueError("usable_segment_count must not be negative")

    def as_csv_dict(self) -> dict[str, str | int]:
        """Return values keyed by the stable report schema."""

        values = asdict(self)
        values["source"] = self.source.value
        return values


def render_access_report_csv(rows: Iterable[AccessReportRow]) -> str:
    """Render report rows as CSV text, including a header for empty reports."""

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(row.as_csv_dict() for row in rows)
    return output.getvalue()


def write_access_report_csv(
    rows: Iterable[AccessReportRow], output_path: str | Path
) -> Path:
    """Write a CSV report; this function never reads or writes media."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_access_report_csv(rows), encoding="utf-8")
    return path
