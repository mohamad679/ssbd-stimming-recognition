import json
from pathlib import Path
import subprocess
import sys

import pytest

import ssbd_behavior.evaluation.provenance as provenance_module
from ssbd_behavior.evaluation import (
    build_artifact_records,
    read_artifact_manifest,
    sha256_file,
    write_artifact_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "build_artifact_manifest.py"


def test_sha256_file_matches_expected_digest(tmp_path) -> None:
    artifact_path = tmp_path / "feature_table.csv"
    artifact_path.write_text("alpha,beta\n1,2\n", encoding="utf-8")

    assert (
        sha256_file(artifact_path)
        == "09a0eff8b13a55262d9a11d1d224a94cfdecaec58b2a72b947fd6b1cb123c46d"
    )


def test_build_artifact_records_include_size_and_type(tmp_path) -> None:
    first_path = tmp_path / "z_report.csv"
    second_path = tmp_path / "a_report.csv"
    first_path.write_text("value\n1\n", encoding="utf-8")
    second_path.write_text("value\n2\n", encoding="utf-8")

    records = build_artifact_records(
        [first_path, second_path],
        artifact_type="evaluation_report",
        notes="synthetic-only",
    )

    assert [record.path for record in records] == sorted(record.path for record in records)
    assert [record.size_bytes for record in records] == [8, 8]
    assert all(record.artifact_type == "evaluation_report" for record in records)
    assert all(record.notes == "synthetic-only" for record in records)


def test_manifest_write_read_roundtrip_is_deterministic(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        provenance_module,
        "_generated_utc_timestamp",
        lambda: "2026-07-05T00:00:00Z",
    )
    first_path = tmp_path / "b_features.csv"
    second_path = tmp_path / "a_keypoints.csv"
    first_path.write_text("value\n1\n", encoding="utf-8")
    second_path.write_text("value\n2\n", encoding="utf-8")
    records = build_artifact_records(
        [first_path, second_path],
        artifact_type="feature_table",
    )
    first_manifest = tmp_path / "manifest.json"
    second_manifest = tmp_path / "manifest_second.json"

    write_artifact_manifest(first_manifest, records)
    manifest = read_artifact_manifest(first_manifest)
    write_artifact_manifest(second_manifest, records)

    assert manifest["schema_version"] == "1.0"
    assert manifest["generated_utc"] == "2026-07-05T00:00:00Z"
    assert manifest["records"] == records
    assert first_manifest.read_text(encoding="utf-8") == second_manifest.read_text(
        encoding="utf-8"
    )


def test_cli_dry_run_writes_nothing(tmp_path) -> None:
    keypoints_path = tmp_path / "keypoints.csv"
    feature_path = tmp_path / "features.csv"
    output_path = tmp_path / "tmp-provenance-test-manifest.json"
    repo_sentinel = REPOSITORY_ROOT / output_path.name
    keypoints_path.write_text("frame,x,y\n0,0.1,0.2\n", encoding="utf-8")
    feature_path.write_text("window,label\n0,1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(feature_path),
            str(keypoints_path),
            "--artifact-type",
            "feature_table",
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Dry run: would write 2 feature_table artifact record(s)" in result.stdout
    assert not output_path.exists()
    assert not repo_sentinel.exists()


def test_cli_execute_writes_manifest(tmp_path) -> None:
    report_path = tmp_path / "evaluation.csv"
    output_path = tmp_path / "tmp-provenance-test-execute.json"
    repo_sentinel = REPOSITORY_ROOT / output_path.name
    report_path.write_text("metric,value\nauroc,0.75\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(report_path),
            "--artifact-type",
            "evaluation_report",
            "--output",
            str(output_path),
            "--execute",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Wrote artifact manifest with 1 record(s)" in result.stdout
    assert output_path.exists()
    assert not repo_sentinel.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["records"][0]["artifact_type"] == "evaluation_report"
    assert payload["records"][0]["size_bytes"] == report_path.stat().st_size


@pytest.mark.parametrize("extension", [".mp4", ".pth"])
def test_cli_rejects_suspicious_media_or_model_extensions(
    tmp_path, extension: str
) -> None:
    suspicious_path = tmp_path / f"suspicious{extension}"
    suspicious_path.write_text("synthetic placeholder\n", encoding="utf-8")
    output_path = tmp_path / "must-not-exist.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(suspicious_path),
            "--artifact-type",
            "keypoint_csv",
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing suspicious raw-media or model-artifact path" in result.stderr
    assert not output_path.exists()
