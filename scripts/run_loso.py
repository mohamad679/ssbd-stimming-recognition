#!/usr/bin/env python3
"""Run deterministic leave-one-group-out baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from run_baselines import _read_numeric_feature_table  # noqa: E402
from ssbd_behavior.evaluation import (  # noqa: E402
    FoldMetricRow,
    binary_classification_metrics,
    format_fold_metrics,
    format_metric_summary,
    summarize_fold_metrics,
)
from ssbd_behavior.models import (  # noqa: E402
    predict_scores,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from ssbd_behavior.validation import (  # noqa: E402
    LOSOFoldStatus,
    leave_one_group_out_indices,
    validate_loso_fold,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline classifiers with leave-one-group-out validation."
    )
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the table and LOSO folds without fitting classifiers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X, y, groups, feature_names = _read_numeric_feature_table(
        args.feature_csv,
        group_column=args.group_column,
        label_column=args.label_column,
    )
    splits = leave_one_group_out_indices(groups)
    print(
        f"Validated {len(y)} rows, {len(set(groups.tolist()))} groups, "
        f"and {len(feature_names)} numeric features."
    )

    statuses = _collect_fold_statuses(y, groups, splits)
    print("\nLOSO fold status:")
    print(_format_fold_statuses(statuses))
    _print_unavailable_notes(statuses)
    if args.dry_run:
        print(f"\nDry run: validated {len(splits)} leave-one-group-out folds; no models fitted.")
        return 0

    trainers = (
        ("logistic_regression", train_logistic_regression_baseline),
        ("random_forest", train_random_forest_baseline),
    )
    rows: list[FoldMetricRow] = []
    for fold_number, ((train_indices, test_indices), status) in enumerate(
        zip(splits, statuses, strict=True),
        start=1,
    ):
        for model_name, trainer in trainers:
            if not status.train_has_two_classes:
                rows.append(
                    FoldMetricRow(
                        fold=fold_number,
                        model_name=model_name,
                        auroc=None,
                        auprc=None,
                        brier_score=None,
                        ece=None,
                        n_test=status.n_test,
                        n_positive=status.n_test_positive,
                        n_negative=status.n_test_negative,
                        groups_tested=(status.group,),
                    )
                )
                continue

            model = trainer(X[train_indices], y[train_indices])
            scores = predict_scores(model, X[test_indices])
            metrics = binary_classification_metrics(y[test_indices], scores)
            rows.append(
                FoldMetricRow(
                    fold=fold_number,
                    model_name=model_name,
                    auroc=metrics["auroc"],
                    auprc=metrics["auprc"],
                    brier_score=metrics["brier_score"],
                    ece=metrics["ece"],
                    n_test=status.n_test,
                    n_positive=status.n_test_positive,
                    n_negative=status.n_test_negative,
                    groups_tested=(status.group,),
                )
            )

    print("\nFold metrics:")
    print(format_fold_metrics(rows))
    print("\nMetric summaries:")
    print(format_metric_summary(summarize_fold_metrics(rows)))
    return 0


def _collect_fold_statuses(
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[LOSOFoldStatus]:
    statuses: list[LOSOFoldStatus] = []
    for train_indices, test_indices in splits:
        held_out_groups = tuple(sorted(set(str(group) for group in groups[test_indices])))
        if len(held_out_groups) != 1:
            raise ValueError("each LOSO fold must hold out exactly one group")
        statuses.append(
            validate_loso_fold(y[train_indices], y[test_indices], held_out_groups[0])
        )
    return statuses


def _format_fold_statuses(statuses: Sequence[LOSOFoldStatus]) -> str:
    header = (
        "fold held_out_group train_has_two_classes test_has_two_classes "
        "n_train n_test n_pos n_neg"
    )
    lines = [header]
    for fold_number, status in enumerate(statuses, start=1):
        lines.append(
            f"{fold_number:>4} {status.group:<14} "
            f"{_format_bool(status.train_has_two_classes):<21} "
            f"{_format_bool(status.test_has_two_classes):<20} "
            f"{status.n_train:>7} {status.n_test:>6} "
            f"{status.n_test_positive:>5} {status.n_test_negative:>5}"
        )
    return "\n".join(lines)


def _print_unavailable_notes(statuses: Sequence[LOSOFoldStatus]) -> None:
    train_unavailable = [status.group for status in statuses if not status.train_has_two_classes]
    test_unavailable = [status.group for status in statuses if not status.test_has_two_classes]
    if train_unavailable:
        print(
            "\nTraining-single-class folds (all metrics unavailable; models not fitted): "
            + ", ".join(train_unavailable)
        )
    if test_unavailable:
        print(
            "\nTest-single-class folds (AUROC/AUPRC unavailable): "
            + ", ".join(test_unavailable)
        )


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
