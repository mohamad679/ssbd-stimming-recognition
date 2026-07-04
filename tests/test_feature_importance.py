import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ssbd_behavior.interpretability import (
    FeatureImportanceRecord,
    extract_model_feature_importance,
    summarize_top_features,
)
from ssbd_behavior.models import (
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "explain_feature_importance.py"
ARTIFACT_EXTENSIONS = {
    ".avi",
    ".bmp",
    ".csv",
    ".gif",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".json",
    ".mkv",
    ".mov",
    ".mp4",
    ".onnx",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".svg",
    ".webp",
}


def _synthetic_binary_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [-2.0, 0.0, 0.4],
            [-1.5, 0.2, -0.2],
            [-1.0, -0.1, 0.1],
            [1.0, 0.1, -0.1],
            [1.5, -0.2, 0.2],
            [2.0, 0.0, -0.4],
        ]
    )
    y = np.asarray([0, 0, 0, 1, 1, 1])
    return X, y


def test_random_forest_feature_importances_extraction_works() -> None:
    X, y = _synthetic_binary_data()
    model = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)

    records = extract_model_feature_importance(
        model,
        ("wrist_periodicity", "head_velocity", "angle_variance"),
        source="random_forest",
    )

    assert len(records) == 3
    assert [record.rank for record in records] == [1, 2, 3]
    assert [record.importance for record in records] == sorted(
        (record.importance for record in records), reverse=True
    )
    assert all(record.source == "random_forest" for record in records)


def test_logistic_regression_coefficient_extraction_uses_absolute_values() -> None:
    X, y = _synthetic_binary_data()
    model = LogisticRegression(solver="liblinear", random_state=7).fit(X, y)

    records = extract_model_feature_importance(
        model,
        ("wrist_periodicity", "head_velocity", "angle_variance"),
        source="logistic_regression",
    )

    expected = sorted(np.abs(model.coef_[0]), reverse=True)
    np.testing.assert_allclose([record.importance for record in records], expected)
    assert all(record.importance >= 0 for record in records)


def test_existing_logistic_pipeline_is_supported() -> None:
    X, y = _synthetic_binary_data()
    model = train_logistic_regression_baseline(X, y)

    records = extract_model_feature_importance(
        model,
        ("wrist_periodicity", "head_velocity", "angle_variance"),
        source="logistic_regression",
    )

    assert len(records) == X.shape[1]
    assert records[0].rank == 1


def test_unsupported_model_raises_clear_error() -> None:
    X, y = _synthetic_binary_data()
    model = DummyClassifier(strategy="most_frequent").fit(X, y)

    with pytest.raises(ValueError, match="unsupported model"):
        extract_model_feature_importance(model, ("a", "b", "c"), source="dummy")


def test_feature_count_mismatch_raises_clear_error() -> None:
    X, y = _synthetic_binary_data()
    model = LogisticRegression(solver="liblinear").fit(X, y)

    with pytest.raises(ValueError, match="feature count mismatch"):
        extract_model_feature_importance(model, ("a", "b"), source="linear")


def test_deterministic_sorting_breaks_ties_by_feature_name() -> None:
    class TiedImportanceModel:
        feature_importances_ = np.asarray([0.5, 0.2, 0.5])

    records = extract_model_feature_importance(
        TiedImportanceModel(),
        ("zeta", "middle", "alpha"),
        source="synthetic",
    )

    assert [record.feature for record in records] == ["alpha", "zeta", "middle"]
    assert [record.rank for record in records] == [1, 2, 3]


def test_summarize_top_features_respects_top_k() -> None:
    records = [
        FeatureImportanceRecord("c", 0.1, "test", 3),
        FeatureImportanceRecord("b", 0.8, "test", 2),
        FeatureImportanceRecord("a", 0.8, "test", 1),
    ]

    summary = summarize_top_features(records, top_k=2)

    assert [record.feature for record in summary] == ["a", "b"]
    assert summarize_top_features(records, top_k=0) == []


def test_cli_dry_run_writes_nothing(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    output_path = tmp_path / "importance.json"
    _write_synthetic_feature_csv(csv_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(csv_path), "--output", str(output_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Model-native exploratory explanation only" in result.stdout
    assert "non-causal" in result.stdout
    assert "not diagnostic or screening evidence" in result.stdout
    assert "Read-only mode" in result.stdout
    assert not output_path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["features.csv"]


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_cli_execute_writes_report_only_to_tmp_path(tmp_path, model_name) -> None:
    csv_path = tmp_path / "features.csv"
    output_path = tmp_path / f"{model_name}.json"
    _write_synthetic_feature_csv(csv_path)
    before_repo_artifacts = _repo_artifact_paths()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(csv_path),
            "--model",
            model_name,
            "--top-k",
            "2",
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
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["model"] == model_name
    assert len(payload["features"]) == 2
    assert "not diagnostic or screening evidence" in payload["notice"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "features.csv",
        f"{model_name}.json",
    ]
    assert _repo_artifact_paths() == before_repo_artifacts


@pytest.mark.parametrize(
    "suffix",
    [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".pkl",
        ".joblib",
        ".pt",
        ".pth",
    ],
)
def test_cli_rejects_suspicious_media_image_and_model_extensions(
    tmp_path, suffix
) -> None:
    input_path = tmp_path / f"suspicious{suffix}"
    input_path.write_text("not a numeric feature table", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(input_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "media/image/model inputs are not supported" in result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [input_path.name]


def test_cli_reuses_baseline_trainers_without_config_override() -> None:
    spec = importlib.util.spec_from_file_location("explain_feature_importance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous

    assert module.MODEL_TRAINERS["logistic_regression"] is train_logistic_regression_baseline
    assert module.MODEL_TRAINERS["random_forest"] is train_random_forest_baseline


def _write_synthetic_feature_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "video_id",
                "window_start_s",
                "window_end_s",
                "label",
                "wrist_periodicity",
                "head_velocity",
                "angle_variance",
            ),
        )
        writer.writeheader()
        for group_index in range(6):
            for label in (0, 1):
                writer.writerow(
                    {
                        "video_id": f"synthetic-{group_index}",
                        "window_start_s": label,
                        "window_end_s": label + 1,
                        "label": label,
                        "wrist_periodicity": label + group_index * 0.01,
                        "head_velocity": label * 0.8 - group_index * 0.01,
                        "angle_variance": (group_index % 2) * 0.2 + label * 0.1,
                    }
                )


def _repo_artifact_paths() -> set[Path]:
    return {
        path.relative_to(REPOSITORY_ROOT)
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in ARTIFACT_EXTENSIONS
    }
