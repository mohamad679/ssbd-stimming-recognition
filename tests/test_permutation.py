import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np

from ssbd_behavior.evaluation import auroc
from ssbd_behavior.models import (
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from ssbd_behavior.validation import (
    group_kfold_indices,
    leave_one_group_out_indices,
    permutation_test_score,
    shuffle_labels_within_groups,
)


def _synthetic_grouped_binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [-2.0, -0.1],
            [2.0, 0.1],
            [-1.8, -0.2],
            [1.8, 0.2],
            [-1.6, -0.1],
            [1.6, 0.1],
            [-1.4, -0.2],
            [1.4, 0.2],
            [-1.2, -0.1],
            [1.2, 0.1],
            [-1.0, -0.2],
            [1.0, 0.2],
        ],
        dtype=float,
    )
    y = np.asarray([0, 1] * 6, dtype=int)
    groups = np.asarray(
        [f"synthetic-{group_index}" for group_index in range(6) for _ in range(2)],
        dtype=object,
    )
    return X, y, groups


def test_shuffle_labels_within_groups_preserves_group_multisets() -> None:
    labels = np.asarray([0, 1, 0, 1, 1, 0, 0, 1], dtype=int)
    groups = np.asarray(["a", "a", "a", "b", "b", "c", "c", "c"], dtype=object)

    shuffled = shuffle_labels_within_groups(labels, groups, random_state=7)

    for group in sorted(set(groups.tolist())):
        original_group_labels = np.sort(labels[groups == group])
        shuffled_group_labels = np.sort(shuffled[groups == group])
        np.testing.assert_array_equal(shuffled_group_labels, original_group_labels)


def test_shuffle_labels_within_groups_is_deterministic_for_fixed_random_state() -> None:
    labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=int)
    groups = np.asarray(["g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3"], dtype=object)

    first = shuffle_labels_within_groups(labels, groups, random_state=23)
    second = shuffle_labels_within_groups(labels, groups, random_state=23)

    np.testing.assert_array_equal(first, second)


def test_permutation_test_score_returns_scores_and_bounded_p_value() -> None:
    X, y, groups = _synthetic_grouped_binary_data()

    result = permutation_test_score(
        X,
        y,
        groups,
        model_trainer=train_logistic_regression_baseline,
        scorer=auroc,
        n_permutations=9,
        random_state=11,
        splits=group_kfold_indices(groups, n_splits=3),
    )

    assert isinstance(result.observed_score, float)
    assert len(result.permutation_scores) == 9
    assert all(isinstance(score, float) for score in result.permutation_scores)
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_permutations == 9
    assert result.scoring_name == "auroc"
    assert result.model_name == "train_logistic_regression_baseline"


def test_permutation_test_honestly_tracks_unavailable_folds() -> None:
    X = np.asarray(
        [
            [-2.0, -0.1],
            [-1.8, -0.2],
            [2.0, 0.1],
            [1.8, 0.2],
            [-0.2, -0.1],
            [0.2, 0.1],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1, 0, 1], dtype=int)
    groups = np.asarray(["holdout-a", "holdout-a", "holdout-b", "holdout-b", "holdout-c", "holdout-c"])

    result = permutation_test_score(
        X,
        y,
        groups,
        model_trainer=train_logistic_regression_baseline,
        scorer=auroc,
        n_permutations=5,
        random_state=5,
        splits=leave_one_group_out_indices(groups),
    )

    assert result.n_scored_folds == 1
    assert result.n_unavailable_folds == 2
    assert 0.0 <= result.p_value <= 1.0


def test_permutation_script_runs_on_synthetic_csv_without_artifacts(tmp_path) -> None:
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
                        "window_start_s": float(label),
                        "window_end_s": float(label + 1),
                        "label": label,
                        "wrist_periodicity": label + group_index * 0.05,
                        "head_velocity": label * 0.7 - group_index * 0.02,
                    }
                )

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_permutation_test.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(csv_path),
            "--model",
            "random_forest",
            "--metric",
            "auroc",
            "--n-splits",
            "3",
            "--n-permutations",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Model: random_forest (train_random_forest_baseline)" in completed.stdout
    assert "Metric: auroc" in completed.stdout
    assert "Observed auroc:" in completed.stdout
    assert "P-value:" in completed.stdout
    assert "Permutations: 8" in completed.stdout
    assert "Note: final defended run should use 1000 permutations." in completed.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == ["features.csv"]


def test_permutation_script_reuses_baseline_trainers_without_shadow_config() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_permutation_test.py"
    spec = importlib.util.spec_from_file_location("run_permutation_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.path.insert(0, str(script.parent))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous

    assert module.MODEL_TRAINERS["logistic_regression"] is train_logistic_regression_baseline
    assert module.MODEL_TRAINERS["random_forest"] is train_random_forest_baseline
