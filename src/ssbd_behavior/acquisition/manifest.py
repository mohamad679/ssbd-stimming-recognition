"""Manifest value objects with no network or download behavior."""

from dataclasses import dataclass
from enum import Enum


class DatasetSource(str, Enum):
    """Dataset sources permitted by the project roadmap."""

    SSBDPLUS = "ssbdplus"
    SSBD_PLUS = "SSBD+"
    ORIGINAL_SSBD = "original SSBD"


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """A single source-manifest entry before any access attempt."""

    video_id: str
    source: DatasetSource
    reference: str
    annotation_status: str = "unknown"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if not self.reference.strip():
            raise ValueError("reference must not be empty")

    def to_access_report_row(self) -> "AccessReportRow":
        """Create an explicitly unattempted report row for this entry."""

        from .reporting import AccessReportRow

        return AccessReportRow(
            attempted_video_id=self.video_id,
            source=self.source,
            url_or_manifest_reference=self.reference,
            download_status="not_attempted",
            annotation_status=self.annotation_status,
            notes=self.notes,
        )
