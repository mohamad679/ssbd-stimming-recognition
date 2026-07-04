import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from ssbd_behavior.models import (
    predict_scores,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)


def _synthetic_binary_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [[-2.0, 0.0], [-1.5, 0.2], [-1.0, -0.1], [1.0, 0.1], [1.5, -0.2], [2.0, 0.0]]
    )
    y = np.asarray([0, 0, 0, 1, 1, 1])
    return X, y


def test_baseline_training_and_scores_are_deterministic_probabilities() -> None:
    X, y = _synthetic_binary_data()

    for trainer in (
        train_logistic_regression_baseline,
        train_random_forest_baseline,
    ):
        first = predict_scores(trainer(X, y, random_state=7), X)
        second = predict_scores(trainer(X, y, random_state=7), X)
        assert first.shape == (len(y),)
        assert np.all((0.0 <= first) & (first <= 1.0))
        np.testing.assert_allclose(first, second)
        assert first[y == 1].mean() > first[y == 0].mean()


def test_baseline_script_runs_on_synthetic_csv_without_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
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
            for label in (0, 1):
                writer.writerow(
                    {
                        "video_id": f"synthetic-{group_index}",
                        "window_start_s": label,
                        "window_end_s": label + 1,
                        "label": label,
                        "wrist_periodicity": label + group_index * 0.01,
                        "head_velocity": label * 0.8 - group_index * 0.01,
                    }
                )

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_baselines.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(csv_path), "--n-splits", "3"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "model=logistic_regression" in completed.stdout
    assert "model=random_forest" in completed.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == ["features.csv"]
