#!/usr/bin/env python3
"""Run leakage-safe D-MS-STF ablations on a numeric feature table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.evaluation import (  # noqa: E402
    evaluate_distilled_ms_stf,
    read_numeric_feature_table,
    write_evaluation_outputs,
)
from ssbd_behavior.models import DistillationConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate D-MS-STF with fold-local teachers, inner cross-fitted "
            "soft labels, GroupKFold, and LOSO."
        )
    )
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--multiscale-prefix", default="ms_")
    parser.add_argument("--group-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--protocol",
        action="append",
        choices=("group_kfold", "loso"),
        help="repeat to select protocols; defaults to both",
    )
    parser.add_argument(
        "--teacher",
        choices=("extra_trees", "random_forest", "hist_gradient_boosting"),
        default="extra_trees",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-calibrated-teacher",
        action="store_true",
        help="add nested group-local sigmoid-calibrated teacher/student variants",
    )
    parser.add_argument(
        "--calibration-splits",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="within-group permutations for D-MS-STF GroupKFold (0 disables)",
    )
    parser.add_argument(
        "--permutation-metric", choices=("auroc", "auprc"), default="auroc"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X, y, groups, feature_names = read_numeric_feature_table(
        args.feature_csv,
        group_column=args.group_column,
        label_column=args.label_column,
    )
    config = DistillationConfig(
        teacher_kind=args.teacher,
        n_estimators=args.n_estimators,
        inner_splits=args.inner_splits,
        alpha=args.alpha,
        temperature=args.temperature,
        calibration_splits=args.calibration_splits,
        random_state=args.random_state,
    )
    result = evaluate_distilled_ms_stf(
        X,
        y,
        groups,
        feature_names,
        group_splits=args.group_splits,
        protocols=tuple(args.protocol or ("group_kfold", "loso")),
        config=config,
        multiscale_prefix=args.multiscale_prefix,
        include_calibrated_teacher=args.include_calibrated_teacher,
        n_permutations=args.n_permutations,
        permutation_metric=args.permutation_metric,
    )
    outputs = write_evaluation_outputs(result, args.output_dir)

    print(
        f"Evaluated {len(y)} rows, {len(set(groups.tolist()))} groups, "
        f"{len(feature_names)} features ({len(result.multiscale_feature_names)} multi-scale)."
    )
    print("protocol method                      auroc      auprc      brier     ece")
    for row in result.aggregates:
        print(
            f"{row.protocol:<10} {row.method:<27} "
            f"{_metric(row.auroc.mean):>10} {_metric(row.auprc.mean):>10} "
            f"{_metric(row.brier_score.mean):>10} {_metric(row.ece.mean):>10}"
        )
    if result.permutation is not None:
        print(
            f"Permutation {result.permutation.metric} p-value: "
            f"{result.permutation.p_value:.6f} "
            f"({result.permutation.n_permutations} permutations)"
        )
    print("Outputs:")
    for path in outputs:
        print(path)
    return 0


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
