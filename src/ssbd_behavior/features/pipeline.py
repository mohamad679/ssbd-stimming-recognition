"""End-to-end feature-table pipeline for numeric pose keypoints."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

from ssbd_behavior.acquisition import SSBDPlusSegment
from ssbd_behavior.pose import PoseKeypoint, validate_video_id

from .engineering import window_feature_dict
from .windowing import WindowSpec, generate_windows_for_video


FEATURE_TABLE_BASE_COLUMNS = (
    "video_id",
    "window_start_s",
    "window_end_s",
    "label",
)
_SAFE_FEATURE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class FeatureRow:
    """One labeled window containing only safe metadata and numeric features."""

    video_id: str
    window_start_s: float
    window_end_s: float
    label: int
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        validate_video_id(self.video_id)
        if not _is_finite_number(self.window_start_s) or self.window_start_s < 0:
            raise ValueError("window_start_s must be a finite, non-negative number")
        if (
            not _is_finite_number(self.window_end_s)
            or self.window_end_s <= self.window_start_s
        ):
            raise ValueError("window_end_s must be finite and greater than window_start_s")
        if (
            isinstance(self.label, bool)
            or not isinstance(self.label, Integral)
            or self.label not in (0, 1)
        ):
            raise ValueError("label must be either 0 or 1")
        if not isinstance(self.features, Mapping):
            raise TypeError("features must be a mapping")

        validated_features: dict[str, float] = {}
        for name, value in self.features.items():
            if (
                not isinstance(name, str)
                or _SAFE_FEATURE_NAME.fullmatch(name) is None
                or name in FEATURE_TABLE_BASE_COLUMNS
            ):
                raise ValueError(f"unsafe or reserved feature name: {name!r}")
            if not _is_finite_number(value):
                raise ValueError(f"feature {name!r} must be a finite numeric value")
            validated_features[name] = float(value)

        object.__setattr__(self, "window_start_s", float(self.window_start_s))
        object.__setattr__(self, "window_end_s", float(self.window_end_s))
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "features", MappingProxyType(validated_features))


def build_feature_rows_for_video(
    video_id: str,
    keypoint_rows: Iterable[PoseKeypoint],
    annotations: Iterable[SSBDPlusSegment],
    spec: WindowSpec,
    sample_rate_hz: float,
    minimum_overlap_fraction: float = 0.5,
) -> list[FeatureRow]:
    """Build one feature row per complete window for a single video.

    Duration is inferred from the final matching frame as one sample interval
    beyond its timestamp. Rows from other videos are discarded before duration
    inference and feature computation.
    """

    validate_video_id(video_id)
    if not _is_finite_number(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite number")

    video_rows: list[PoseKeypoint] = []
    for row in keypoint_rows:
        if not isinstance(row, PoseKeypoint):
            raise TypeError("keypoint_rows must contain PoseKeypoint instances")
        if row.video_id == video_id:
            video_rows.append(row)

    if not video_rows:
        raise ValueError(f"no keypoint rows found for video_id {video_id!r}")

    duration_s = max(row.timestamp_s for row in video_rows) + 1.0 / float(
        sample_rate_hz
    )
    windows = generate_windows_for_video(
        video_id=video_id,
        duration_s=duration_s,
        annotations=annotations,
        spec=spec,
        minimum_overlap_fraction=minimum_overlap_fraction,
    )

    feature_rows: list[FeatureRow] = []
    for window in windows:
        rows_in_window = [
            row
            for row in video_rows
            if window.start_s <= row.timestamp_s < window.end_s
        ]
        feature_rows.append(
            FeatureRow(
                video_id=window.video_id,
                window_start_s=window.start_s,
                window_end_s=window.end_s,
                label=window.label,
                features=window_feature_dict(rows_in_window, sample_rate_hz),
            )
        )
    return feature_rows


def write_feature_table_csv(
    path: Path | str, rows: Iterable[FeatureRow]
) -> None:
    """Write feature rows to CSV with a stable, numeric feature schema."""

    feature_rows = list(rows)
    for row in feature_rows:
        if not isinstance(row, FeatureRow):
            raise TypeError("rows must contain FeatureRow instances")

    feature_names = tuple(feature_rows[0].features) if feature_rows else ()
    for row in feature_rows[1:]:
        if tuple(row.features) != feature_names:
            raise ValueError("all feature rows must have the same ordered feature names")

    csv_path = Path(path)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = FEATURE_TABLE_BASE_COLUMNS + feature_names
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows:
            writer.writerow(
                {
                    "video_id": row.video_id,
                    "window_start_s": row.window_start_s,
                    "window_end_s": row.window_end_s,
                    "label": row.label,
                    **row.features,
                }
            )


def read_feature_table_csv(path: Path | str) -> list[FeatureRow]:
    """Read and validate a feature table produced by this module."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("feature table CSV columns must be unique")
        if fieldnames[: len(FEATURE_TABLE_BASE_COLUMNS)] != FEATURE_TABLE_BASE_COLUMNS:
            raise ValueError(
                "feature table CSV must begin with columns: "
                + ", ".join(FEATURE_TABLE_BASE_COLUMNS)
            )
        feature_names = fieldnames[len(FEATURE_TABLE_BASE_COLUMNS) :]

        rows: list[FeatureRow] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                if str(label) != row["label"]:
                    raise ValueError("label must be an integer")
                rows.append(
                    FeatureRow(
                        video_id=row["video_id"],
                        window_start_s=float(row["window_start_s"]),
                        window_end_s=float(row["window_end_s"]),
                        label=label,
                        features={name: float(row[name]) for name in feature_names},
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid feature table CSV row {row_number}: {exc}"
                ) from exc
    return rows


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
