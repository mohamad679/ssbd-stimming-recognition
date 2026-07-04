#!/usr/bin/env python3
"""Summarize baseline-model feature importance from a numeric CSV table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssbd_behavior.interpretability import (  # noqa: E402
    FeatureImportanceRecord,
    extract_model_feature_importance,
    summarize_top_features,
)
from ssbd_behavior.models import (  # noqa: E402
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)


MODEL_TRAINERS = {
    "logistic_regression": train_logistic_regression_baseline,
    "random_forest": train_random_forest_baseline,
}

PIPELINE_METADATA_COLUMNS = frozenset(
    {"video_id", "window_start_s", "window_end_s", "label"}
)

SUSPICIOUS_EXTENSIONS = frozenset(
    {
        ".avi",
        ".jpeg",
        ".jpg",
        ".joblib",
        ".mkv",
        ".mov",
        ".mp4",
        ".pkl",
        ".png",
        ".pt",
        ".pth",
        ".webp",
    }
)

EXPLANATION_NOTICE = (
    "Model-native exploratory explanation only; importances are non-causal and "
    "are not diagnostic or screening evidence. No raw frames, images, or videos "
    "are read."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize model-native exploratory feature importance from a numeric "
            "feature-table CSV. This command does not read media or save models."
        )
    )
    parser.add_argument("feature_csv", type=Path, help="numeric feature-table CSV")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="video_id")
    parser.add_argument(
        "--model",
        choices=tuple(sorted(MODEL_TRAINERS)),
        default="logistic_regression",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional .json or .csv explanation report path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write --output; omitted by default for read-only mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = _validate_input_path(args.feature_csv)
        output_path = (
            _validate_output_path(args.output) if args.output is not None else None
        )
        if args.execute and output_path is None:
            raise ValueError("--execute requires --output")
        X, y, feature_names = _read_numeric_feature_table(
            input_path,
            group_column=args.group_column,
            label_column=args.label_column,
        )
        if np.unique(y).size < 2:
            raise ValueError("feature table labels must contain both classes 0 and 1")
        model = MODEL_TRAINERS[args.model](X, y)
        records = extract_model_feature_importance(
            model,
            feature_names,
            source=args.model,
        )
        top_records = summarize_top_features(records, top_k=args.top_k)
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(EXPLANATION_NOTICE)
    for record in top_records:
        print(f"{record.rank:>2}. {record.feature}: {record.importance:.12g}")

    if not args.execute:
        if output_path is not None:
            print(f"Read-only mode: use --execute to write {output_path}")
        return 0

    try:
        _write_report(output_path, args.model, top_records)
    except OSError as exc:
        print(f"error: could not write explanation report: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote explanation report: {output_path}")
    return 0


def _validate_input_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in SUSPICIOUS_EXTENSIONS:
        raise ValueError(f"media/image/model inputs are not supported: {path}")
    if suffix != ".csv":
        raise ValueError(f"input path must be a numeric feature-table .csv file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"feature-table CSV does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"feature-table CSV path must be a file: {path}")
    return path


def _validate_output_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in SUSPICIOUS_EXTENSIONS:
        raise ValueError(f"media/image/model outputs are not supported: {path}")
    if suffix not in {".csv", ".json"}:
        raise ValueError(f"output path must have a .json or .csv extension: {path}")
    return path


def _read_numeric_feature_table(
    path: Path, *, group_column: str, label_column: str
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
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
        for row_number, row in enumerate(reader, start=2):
            try:
                group = row[group_column].strip()
                if not group:
                    raise ValueError("group is missing")
                label_text = row[label_column]
                label = int(label_text)
                if str(label) != label_text or label not in (0, 1):
                    raise ValueError("label must be exactly 0 or 1")
                numeric_row = [float(row[name]) for name in feature_names]
                if not all(math.isfinite(value) for value in numeric_row):
                    raise ValueError("features must be finite")
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid feature table row {row_number}: {exc}") from exc
            labels.append(label)
            features.append(numeric_row)

    if not features:
        raise ValueError("feature table must contain at least one data row")
    return (
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=int),
        feature_names,
    )


def _write_report(
    output_path: Path,
    model_name: str,
    records: Sequence[FeatureImportanceRecord],
) -> None:
    if output_path.suffix.lower() == ".json":
        payload = {
            "notice": EXPLANATION_NOTICE,
            "model": model_name,
            "features": [asdict(record) for record in records],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "model",
            "feature",
            "importance",
            "source",
            "rank",
            "notes",
            "notice",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model": model_name,
                    **asdict(record),
                    "notice": EXPLANATION_NOTICE,
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
