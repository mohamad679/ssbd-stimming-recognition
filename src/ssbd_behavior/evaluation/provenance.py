"""Provenance helpers for numeric artifact manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0"
_READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Provenance metadata for one derived artifact file."""

    path: str
    sha256: str
    size_bytes: int
    artifact_type: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("path must not be empty")
        if not self.sha256.strip():
            raise ValueError("sha256 must not be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not self.artifact_type.strip():
            raise ValueError("artifact_type must not be empty")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest for one file by streaming its content."""

    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"artifact file does not exist: {candidate}")
    if candidate.is_dir():
        raise IsADirectoryError(f"artifact path must be a file, not a directory: {candidate}")

    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(_READ_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_records(
    paths: Iterable[str | Path], *, artifact_type: str, notes: str | None = None
) -> tuple[ArtifactRecord, ...]:
    """Build deterministic provenance records for explicit artifact files."""

    if not artifact_type.strip():
        raise ValueError("artifact_type must not be empty")

    records = []
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.exists():
            raise FileNotFoundError(f"artifact file does not exist: {candidate}")
        if candidate.is_dir():
            raise IsADirectoryError(
                f"artifact path must be a file, not a directory: {candidate}"
            )
        records.append(
            ArtifactRecord(
                path=_normalize_record_path(candidate),
                sha256=sha256_file(candidate),
                size_bytes=candidate.stat().st_size,
                artifact_type=artifact_type,
                notes=notes,
            )
        )
    return tuple(sorted(records, key=lambda record: record.path))


def write_artifact_manifest(path: str | Path, records: Iterable[ArtifactRecord]) -> None:
    """Write a deterministic JSON manifest for explicit artifact records."""

    manifest_path = Path(path)
    if manifest_path.exists() and manifest_path.is_dir():
        raise IsADirectoryError(
            f"manifest path must be a file, not a directory: {manifest_path}"
        )

    ordered_records = tuple(sorted(records, key=lambda record: record.path))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _generated_utc_timestamp(),
        "records": [asdict(record) for record in ordered_records],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_artifact_manifest(path: str | Path) -> dict[str, object]:
    """Read a JSON artifact manifest written by :func:`write_artifact_manifest`."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"artifact manifest does not exist: {manifest_path}")
    if manifest_path.is_dir():
        raise IsADirectoryError(
            f"artifact manifest path must be a file, not a directory: {manifest_path}"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("artifact manifest records must be a list")
    return {
        "schema_version": payload.get("schema_version"),
        "generated_utc": payload.get("generated_utc"),
        "records": tuple(ArtifactRecord(**record) for record in raw_records),
    }


def _generated_utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _normalize_record_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
