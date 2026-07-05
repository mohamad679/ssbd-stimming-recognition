import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ssbd_behavior.evaluation import (
    evaluate_distilled_ms_stf,
    write_evaluation_outputs,
)
from ssbd_behavior.models import (
    DistillationConfig,
    cross_fitted_teacher_soft_labels,
    predict_scores,
    train_teacher,
)


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = []
    labels = []
    groups = []
    for group_index in range(6):
        for sample_index, label in enumerate((0, 1, 0, 1)):
            offset = group_index * 0.02 + sample_index * 0.005
            rows.append(
                [
                    label + offset,
                    label * 0.7 - offset,
                    label * 1.2 + offset,
                    label * 0.4 + group_index * 0.01,
                ]
            )
            labels.append(label)
            groups.append(f"synthetic-{group_index}")
    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(groups, dtype=object),
        ("baseline_motion", "baseline_periodicity", "ms_1s__energy", "ms_4s__energy"),
    )


def _config() -> DistillationConfig:
    return DistillationConfig(n_estimators=8, inner_splits=2, random_state=9)


def test_cross_fitted_soft_labels_never_train_on_validation_groups() -> None:
    X, y, groups, _ = _synthetic_data()

    result = cross_fitted_teacher_soft_labels(X, y, groups, config=_config())

    assert result.probabilities.shape == y.shape
    assert np.all((0.0 <= result.probabilities) & (result.probabilities <= 1.0))
    validation_indices = []
    for fold in result.folds:
        assert set(fold.train_groups).isdisjoint(fold.validation_groups)
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        validation_indices.extend(fold.validation_indices)
    assert sorted(validation_indices) == list(range(len(y)))


def test_optional_teacher_calibration_stays_group_local() -> None:
    X, y, groups, _ = _synthetic_data()
    config = DistillationConfig(
        n_estimators=4,
        inner_splits=2,
        calibrate_teacher=True,
        calibration_splits=2,
        random_state=3,
    )

    teacher = train_teacher(X, y, groups=groups, config=config)
    scores = predict_scores(teacher, X)

    assert scores.shape == y.shape
    assert np.all((0.0 <= scores) & (scores <= 1.0))


def test_evaluation_is_fold_local_and_writes_stable_output_schema(tmp_path) -> None:
    X, y, groups, names = _synthetic_data()

    result = evaluate_distilled_ms_stf(
        X,
        y,
        groups,
        names,
        group_splits=3,
        config=_config(),
        n_permutations=2,
    )

    assert {row.protocol for row in result.aggregates} == {"group_kfold", "loso"}
    assert {row.method for row in result.aggregates} == {
        "current_logistic_baseline",
        "current_random_forest",
        "ms_stf_only",
        "teacher_only",
        "student_hard",
        "d_ms_stf",
    }
    assert result.permutation is not None
    assert 0.0 <= result.permutation.p_value <= 1.0
    for audit in result.audits:
        assert set(audit.outer_train_groups).isdisjoint(audit.outer_test_groups)
        for inner in audit.inner_folds:
            assert set(inner.train_groups).isdisjoint(inner.validation_groups)
            assert set(audit.outer_test_groups).isdisjoint(inner.train_groups)
            assert set(audit.outer_test_groups).isdisjoint(inner.validation_groups)

    paths = write_evaluation_outputs(result, tmp_path / "outputs")
    assert [path.name for path in paths] == [
        "fold_metrics.csv",
        "aggregate_metrics.csv",
        "report.json",
    ]
    report = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["method"] == "D-MS-STF"
    assert report["leakage_audits"]
    with paths[1].open(encoding="utf-8", newline="") as stream:
        columns = tuple(csv.DictReader(stream).fieldnames or ())
    assert "auroc_mean" in columns
    assert "auprc_mean" in columns
    assert "brier_score_mean" in columns
    assert "ece_mean" in columns
    assert "permutation_p_value" in columns


def test_stage_f_runner_works_on_synthetic_numeric_csv(tmp_path) -> None:
    X, y, groups, names = _synthetic_data()
    csv_path = tmp_path / "features.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("video_id", "label", *names))
        writer.writeheader()
        for row, label, group in zip(X, y, groups, strict=True):
            writer.writerow(
                {"video_id": group, "label": int(label)}
                | dict(zip(names, row, strict=True))
            )

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_distilled_ms_stf.py"
    output_dir = tmp_path / "stage-f-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--protocol",
            "group_kfold",
            "--group-splits",
            "3",
            "--inner-splits",
            "2",
            "--n-estimators",
            "6",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "d_ms_stf" in completed.stdout
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "aggregate_metrics.csv",
        "fold_metrics.csv",
        "report.json",
    ]
