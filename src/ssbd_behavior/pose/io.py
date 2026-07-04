"""CSV persistence for numeric pose keypoint rows."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from .keypoints import PoseKeypoint


KEYPOINT_CSV_COLUMNS = (
    "video_id",
    "frame_index",
    "timestamp_s",
    "landmark_index",
    "x",
    "y",
    "z",
    "confidence",
)


def write_keypoints_csv(path: Path | str, rows: Iterable[PoseKeypoint]) -> None:
    """Write validated numeric keypoint rows to a UTF-8 CSV file."""

    csv_path = Path(path)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=KEYPOINT_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, PoseKeypoint):
                raise TypeError("rows must contain PoseKeypoint instances")
            writer.writerow(
                {
                    "video_id": row.video_id,
                    "frame_index": row.frame_index,
                    "timestamp_s": row.timestamp_s,
                    "landmark_index": row.landmark_index,
                    "x": row.x,
                    "y": row.y,
                    "z": "" if row.z is None else row.z,
                    "confidence": row.confidence,
                }
            )


def read_keypoints_csv(path: Path | str) -> list[PoseKeypoint]:
    """Read and validate keypoint rows from a CSV produced by this module."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != KEYPOINT_CSV_COLUMNS:
            raise ValueError(
                "keypoint CSV columns must be exactly: "
                + ", ".join(KEYPOINT_CSV_COLUMNS)
            )

        rows: list[PoseKeypoint] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                z_value = row["z"]
                rows.append(
                    PoseKeypoint(
                        video_id=row["video_id"],
                        frame_index=int(row["frame_index"]),
                        timestamp_s=float(row["timestamp_s"]),
                        landmark_index=int(row["landmark_index"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=None if z_value == "" else float(z_value),
                        confidence=float(row["confidence"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid keypoint CSV row {row_number}: {exc}"
                ) from exc
    return rows
