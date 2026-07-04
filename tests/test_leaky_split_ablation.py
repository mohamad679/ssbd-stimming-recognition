import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "diagnostics" / "leaky_split_ablation.py"
)
BASELINE_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_baselines.py"


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
            ),
        )
        writer.writeheader()
        for group_index in range(6):
            for window_index in range(4):
                label = window_index % 2
                writer.writerow(
                    {
                        "video_id": f"synthetic-{group_index}",
                        "window_start_s": window_index,
                        "window_end_s": window_index + 1,
                        "label": label,
                        "wrist_periodicity": label + group_index * 0.01,
                        "head_velocity": label * 0.8 - group_index * 0.01,
                    }
                )


def _load_diagnostic_module():
    spec = importlib.util.spec_from_file_location("leaky_split_ablation", DIAGNOSTIC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_runs_with_warning_metrics_and_no_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_synthetic_feature_csv(csv_path)

    completed = subprocess.run(
        [sys.executable, str(DIAGNOSTIC_SCRIPT), str(csv_path), "--n-splits", "3"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "WARNING: THIS IS A LEAKY DIAGNOSTIC ABLATION. "
        "DO NOT REPORT AS VALID PERFORMANCE."
    ) in completed.stdout
    assert "This is for leakage sensitivity comparison only." in completed.stdout
    assert "model=logistic_regression" in completed.stdout
    assert "model=random_forest" in completed.stdout
    assert "auroc=" in completed.stdout
    assert "auprc=" in completed.stdout
    assert "brier_score=" in completed.stdout
    assert "ece=" in completed.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == ["features.csv"]


def test_main_baseline_runner_does_not_expose_leaky_flag() -> None:
    completed = subprocess.run(
        [sys.executable, str(BASELINE_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--leaky" not in completed.stdout.lower()


def test_diagnostic_split_can_overlap_groups() -> None:
    diagnostic = _load_diagnostic_module()
    groups = np.repeat([f"synthetic-{index}" for index in range(6)], 4)
    labels = np.tile([0, 1, 0, 1], 6)

    splits = diagnostic.leaky_kfold_indices(labels, n_splits=3)

    assert any(
        set(groups[train_indices]) & set(groups[test_indices])
        for train_indices, test_indices in splits
    )
