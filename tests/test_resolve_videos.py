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
