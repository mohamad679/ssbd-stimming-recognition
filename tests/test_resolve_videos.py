import csv
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "acquisition" / "resolve_videos.py"


def test_default_mode_is_offline_dry_run(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,source,url_or_manifest_reference,annotation_status\n"
        "plus-001,SSBD+,manifest.xml#plus-001,present\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "plus-001,SSBD+,manifest.xml#plus-001,not_attempted" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == [manifest]


def test_execute_is_explicitly_unimplemented():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--execute"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "intentionally not implemented" in result.stderr


def _write_ssbdplus_manifest(path: Path) -> None:
    path.write_text(
        "xml_file_name,youtube_video_url,action_start_time,action_end_time,action_category\n"
        "video-001.xml,https://example.test/watch?v=001,1,4,arm-flapping\n"
        "video-001.xml,https://example.test/watch?v=001,8,12,head-banging\n"
        "video-002.xml,https://example.test/watch?v=002,2,6,spinning\n",
        encoding="utf-8",
    )


def test_ssbdplus_dry_run_aggregates_segments_by_video(tmp_path):
    manifest = tmp_path / "ssbdplus_dataset.csv"
    _write_ssbdplus_manifest(manifest)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    rows = list(csv.DictReader(result.stdout.splitlines()))
    assert len(rows) == 2
    assert {row["attempted_video_id"] for row in rows} == {"video-001", "video-002"}
    assert {row["attempted_video_id"]: row["usable_segment_count"] for row in rows} == {
        "video-001": "2",
        "video-002": "1",
    }
    assert all(row["source"] == "ssbdplus" for row in rows)
    assert all(row["download_status"] == "not_attempted" for row in rows)
    assert all(row["annotation_status"] == "present" for row in rows)
    assert all("video availability not verified" in row["notes"] for row in rows)
    assert result.stderr == ""


def test_ssbdplus_dry_run_writes_output_file(tmp_path):
    manifest = tmp_path / "ssbdplus_dataset.csv"
    output = tmp_path / "reports" / "access.csv"
    _write_ssbdplus_manifest(manifest)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert "wrote access report" in result.stderr
    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_execute_does_not_read_manifest_or_write_output(tmp_path):
    missing_manifest = tmp_path / "missing.csv"
    output = tmp_path / "must-not-exist.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--execute",
            "--manifest",
            str(missing_manifest),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "intentionally not implemented" in result.stderr
    assert not output.exists()
