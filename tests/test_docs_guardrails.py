from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8").lower()


def test_model_card_contains_non_diagnostic_guardrails():
    content = _read("docs/model_card.md")
    assert "not autism diagnosis" in content or "not an autism diagnosis" in content
    assert "not autism screening" in content or "not an autism screening" in content
    assert "not clinical triage" in content
    assert "not surveillance" in content
    assert "not deployment-ready" in content or "deployment-ready decision support" in content


def test_limitations_state_non_deployment_and_non_clinical_status():
    content = _read("docs/limitations.md")
    assert "not deployment-ready" in content
    assert "not clinically validated" in content or "does not provide diagnostic validity" in content
    assert "not an autism diagnosis tool" in content


def test_final_status_reflects_completed_accessible_video_benchmark():
    content = _read("docs/final_project_status.md")
    assert "completed accessible-video ssbd+ benchmark run" in content
    assert "non-diagnostic research scaffold" in content


def test_full_benchmark_report_keeps_non_diagnostic_guardrails():
    content = _read("docs/full_ssbdplus_benchmark_report.md")
    assert "not autism diagnosis" in content or "not an autism diagnosis" in content
    assert "not screening" in content
    assert "not deployment-ready" in content


def test_readme_keeps_non_diagnostic_framing():
    content = _read("README.md")
    assert "not an autism diagnostic or screening tool" in content
    assert "has not been clinically validated" in content


def test_roadmap_excludes_internal_notes():
    content = _read("SSBD_Behavior_Recognition_Roadmap.md")
    assert "realistic time estimate" not in content
    assert "kickoff prompt" not in content
    assert "claude code" not in content


def test_ci_workflow_stays_privacy_safe():
    content = _read(".github/workflows/tests.yml")
    assert "pytest" in content
    assert "compileall" in content
    forbidden_snippets = [
        "yt-dlp",
        "youtube",
        "download",
        "upload-artifact",
        "actions/upload-artifact",
        "mediapipe",
        "opencv",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in content
