import importlib.util
import sys
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "packaging" / "build_ssbd_colab_package.py"


def _load_packager():
    spec = importlib.util.spec_from_file_location("build_ssbd_colab_package", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_ssbd_colab_package_excludes_unsafe_files_and_preserves_layout(tmp_path):
    packager = _load_packager()
    repo_root = tmp_path / "repo"
    metadata_dir = tmp_path / "metadata"
    output_zip = tmp_path / "ssbd_colab_package.zip"

    (repo_root / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (repo_root / "README.md").write_text("readme\n", encoding="utf-8")
    (repo_root / "src" / "ssbd_behavior" / "models").mkdir(parents=True)
    (repo_root / "src" / "ssbd_behavior" / "models" / "__init__.py").write_text(
        "# package\n", encoding="utf-8"
    )
    (repo_root / "src" / "ssbd_behavior" / "data").mkdir(parents=True)
    (repo_root / "src" / "ssbd_behavior" / "data" / "nested.csv").write_text(
        "x\n1\n", encoding="utf-8"
    )
    (repo_root / "data").mkdir()
    (repo_root / "data" / "raw_clip.mp4").write_bytes(b"unsafe")
    (repo_root / "__pycache__").mkdir()
    (repo_root / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
    (repo_root / ".git").mkdir()
    (repo_root / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    metadata_dir.mkdir()
    (metadata_dir / "metadata.csv").write_text("x\n1\n", encoding="utf-8")
    (metadata_dir / "preview.png").write_bytes(b"unsafe")

    members = packager.build_ssbd_colab_package(repo_root, metadata_dir, output_zip)

    assert "ssbd_colab_package/ssbd-stimming-recognition/README.md" in members
    assert (
        "ssbd_colab_package/ssbd-stimming-recognition/src/ssbd_behavior/models/__init__.py"
        in members
    )
    assert "ssbd_colab_package/metadata/metadata.csv" in members
    assert "ssbd_colab_package/ssbd-stimming-recognition/src/ssbd_behavior/data/nested.csv" not in members
    assert not any(member.endswith(".mp4") for member in members)
    assert not any(
        member.startswith("ssbd_colab_package/ssbd-stimming-recognition/data/")
        for member in members
    )
    assert not any(member.startswith("ssbd_colab_package/.git/") for member in members)
    assert not any(member.endswith(".png") for member in members)

    with zipfile.ZipFile(output_zip) as archive:
        assert set(archive.namelist()) == set(members)
