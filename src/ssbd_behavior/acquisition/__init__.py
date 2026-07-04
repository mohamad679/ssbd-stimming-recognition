"""Offline helpers for representing SSBD data-access records."""

from .manifest import DatasetSource, ManifestEntry
from .reporting import AccessReportRow, render_access_report_csv, write_access_report_csv
from .ssbdplus import (
    SSBDPlusSegment,
    load_ssbdplus_csv,
    parse_ssbdplus_xml,
    parse_time_range,
    summarize_segments,
)

__all__ = [
    "AccessReportRow",
    "DatasetSource",
    "ManifestEntry",
    "SSBDPlusSegment",
    "load_ssbdplus_csv",
    "parse_ssbdplus_xml",
    "parse_time_range",
    "render_access_report_csv",
    "summarize_segments",
    "write_access_report_csv",
]
