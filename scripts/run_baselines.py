#!/usr/bin/env python3
"""Run deterministic baselines on a local numeric feature table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.evaluation import binary_classification_metrics  # noqa: E402
from ssbd_behavior.models import (  # noqa: E402
    predict_scores,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from ssbd_behavior.validation import group_kfold_indices  # noqa: E402


PIPELINE_METADATA_COLUMNS = frozenset(
    {"video_id", "window_start_s", "window_end_s", "label"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline classifiers with group-disjoint GroupKFold."
    )
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the table and splits without fitting classifiers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X, y, groups, feature_names = _read_numeric_feature_table(
        args.feature_csv,
        group_column=args.group_column,
        label_column=args.label_column,
    )
    splits = group_kfold_indices(groups, args.n_splits)
    print(
        f"Validated {len(y)} rows, {len(set(groups.tolist()))} groups, "
        f"and {len(feature_names)} numeric features."
    )
    if args.dry_run:
        print(f"Dry run: validated {len(splits)} group-disjoint folds; no models fitted.")
        return 0

    trainers = (
        ("logistic_regression", train_logistic_regression_baseline),
        ("random_forest", train_random_forest_baseline),
    )
    for fold_number, (train_indices, test_indices) in enumerate(splits, start=1):
        if np.unique(y[train_indices]).size < 2:
            raise ValueError(
                f"fold {fold_number} training set has only one class; "
                "baseline fitting is unavailable"
            )
        for model_name, trainer in trainers:
            model = trainer(X[train_indices], y[train_indices])
            scores = predict_scores(model, X[test_indices])
            metrics = binary_classification_metrics(y[test_indices], scores)
            print(
                f"fold={fold_number} model={model_name} "
                + " ".join(
                    f"{name}={'unavailable' if value is None else f'{value:.6f}'}"
                    for name, value in metrics.items()
                )
            )
    return 0


def _read_numeric_feature_table(
    path: Path, *, group_column: str, label_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        if len(columns) != len(set(columns)):
            raise ValueError("feature table columns must be unique")
        for required in (group_column, label_column):
            if required not in columns:
                raise ValueError(f"feature table is missing required column {required!r}")

        excluded = PIPELINE_METADATA_COLUMNS | {group_column, label_column}
        feature_names = tuple(name for name in columns if name not in excluded)
        if not feature_names:
            raise ValueError("feature table must contain at least one feature column")

        features: list[list[float]] = []
        labels: list[int] = []
        groups: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                group = row[group_column].strip()
                if not group:
                    raise ValueError("group is missing")
                label = int(row[label_column])
                if str(label) != row[label_column] or label not in (0, 1):
                    raise ValueError("label must be exactly 0 or 1")
                numeric_row = [float(row[name]) for name in feature_names]
                if not all(math.isfinite(value) for value in numeric_row):
                    raise ValueError("features must be finite")
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid feature table row {row_number}: {exc}") from exc
            groups.append(group)
            labels.append(label)
            features.append(numeric_row)

    if not features:
        raise ValueError("feature table must contain at least one data row")
    return (
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(groups, dtype=object),
        feature_names,
    )


if __name__ == "__main__":
    raise SystemExit(main())
