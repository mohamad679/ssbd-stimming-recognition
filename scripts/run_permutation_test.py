#!/usr/bin/env python3
"""Run a group-aware permutation test on a numeric feature table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from run_baselines import _read_numeric_feature_table  # noqa: E402
from ssbd_behavior.evaluation import auprc, auroc  # noqa: E402
from ssbd_behavior.models import (  # noqa: E402
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from ssbd_behavior.validation import (  # noqa: E402
    group_kfold_indices,
    permutation_test_score,
)


MODEL_TRAINERS = {
    "logistic_regression": train_logistic_regression_baseline,
    "random_forest": train_random_forest_baseline,
}

METRIC_SCORERS = {
    "auprc": auprc,
    "auroc": auroc,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run within-group permutation testing with the same baseline trainer "
            "configuration used for the defended benchmark."
        )
    )
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--model",
        choices=tuple(sorted(MODEL_TRAINERS)),
        default="logistic_regression",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(sorted(METRIC_SCORERS)),
        default="auroc",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help=(
            "number of within-group permutations to run; CLI defaults to 100 for "
            "local safety, while the final defended run should use 1000"
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X, y, groups, feature_names = _read_numeric_feature_table(
        args.feature_csv,
        group_column=args.group_column,
        label_column=args.label_column,
    )
    splits = group_kfold_indices(groups, args.n_splits)
    result = permutation_test_score(
        X,
        y,
        groups,
        model_trainer=MODEL_TRAINERS[args.model],
        scorer=METRIC_SCORERS[args.metric],
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        splits=splits,
    )

    print(
        f"Validated {len(y)} rows, {len(set(groups.tolist()))} groups, "
        f"and {len(feature_names)} numeric features."
    )
    print(
        f"Model: {args.model} ({result.model_name})"
    )
    print(f"Metric: {result.scoring_name}")
    print(f"Observed {result.scoring_name}: {result.observed_score:.6f}")
    print(f"P-value: {result.p_value:.6f}")
    print(f"Permutations: {result.n_permutations}")
    print(f"Scored folds: {result.n_scored_folds}")
    if result.n_unavailable_folds:
        print(
            f"Unavailable folds for {result.scoring_name}: "
            f"{result.n_unavailable_folds}"
        )
    if args.n_permutations < 1000:
        print("Note: final defended run should use 1000 permutations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
