import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ssbd_behavior.validation import (
    LOSOFoldStatus,
    leave_one_group_out_indices,
    validate_loso_fold,
)


def test_leave_one_group_out_creates_one_fold_per_group_without_overlap() -> None:
    groups = np.asarray(["g1", "g1", "g2", "g2", "g3"])

    splits = leave_one_group_out_indices(groups)

    assert len(splits) == 3
    held_out_groups = []
    for train_indices, test_indices in splits:
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        assert len(test_groups) == 1
        assert train_groups.isdisjoint(test_groups)
        held_out_groups.extend(test_groups)
    assert sorted(held_out_groups) == ["g1", "g2", "g3"]


def test_leave_one_group_out_rejects_single_group_input() -> None:
    with pytest.raises(ValueError, match="at least 2 distinct groups"):
        leave_one_group_out_indices(["only-group", "only-group"])


def test_validate_loso_fold_captures_single_class_train_or_test() -> None:
    train_unavailable = validate_loso_fold([0, 0, 0], [0, 1], "group-a")
    test_unavailable = validate_loso_fold([0, 1, 0], [1, 1], "group-b")

    assert train_unavailable == LOSOFoldStatus(
        group="group-a",
        train_has_two_classes=False,
        test_has_two_classes=True,
        n_train=3,
        n_test=2,
        n_test_positive=1,
        n_test_negative=1,
    )
    assert test_unavailable == LOSOFoldStatus(
        group="group-b",
        train_has_two_classes=True,
        test_has_two_classes=False,
        n_train=3,
        n_test=2,
        n_test_positive=2,
        n_test_negative=0,
    )


def test_run_loso_script_runs_on_synthetic_csv_without_artifacts(tmp_path) -> None:
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
        rows = [
            ("holdout-a", 0, 0.05, -0.05),
            ("holdout-a", 0, 0.10, -0.10),
            ("holdout-b", 1, 0.95, 0.85),
            ("holdout-b", 1, 1.00, 0.90),
            ("holdout-c", 0, 0.15, -0.15),
            ("holdout-c", 1, 0.85, 0.75),
        ]
        for group, label, periodicity, velocity in rows:
            writer.writerow(
                {
                    "video_id": group,
                    "window_start_s": 0.0,
                    "window_end_s": 1.0,
                    "label": label,
                    "wrist_periodicity": periodicity,
                    "head_velocity": velocity,
                }
            )

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_loso.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(csv_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "LOSO fold status:" in completed.stdout
    assert "Fold metrics:" in completed.stdout
    assert "Metric summaries:" in completed.stdout
    assert "holdout-a" in completed.stdout
    assert "holdout-b" in completed.stdout
    assert "holdout-c" in completed.stdout
    assert "logistic_regression" in completed.stdout
    assert "random_forest" in completed.stdout
    assert "Test-single-class folds (AUROC/AUPRC unavailable): holdout-a, holdout-b" in completed.stdout
    assert "model=logistic_regression folds=3" in completed.stdout
    assert "model=random_forest folds=3" in completed.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == ["features.csv"]


def test_run_loso_script_exposes_no_leaky_option() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_loso.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "leaky" not in completed.stdout.lower()
