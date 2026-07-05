"""Evaluation and privacy-safe reporting for the D-MS-STF method."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from ssbd_behavior.evaluation.metrics import binary_classification_metrics
from ssbd_behavior.models.baselines import (
    predict_scores,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from ssbd_behavior.models.distilled_ms_stf import (
    DistillationConfig,
    InnerFoldAudit,
    cross_fitted_teacher_soft_labels,
    train_distilled_student,
    train_hard_label_student,
    train_teacher,
)
from ssbd_behavior.validation.loso import leave_one_group_out_indices
from ssbd_behavior.validation.permutation import shuffle_labels_within_groups
from ssbd_behavior.validation.splits import group_kfold_indices


Protocol = Literal["group_kfold", "loso"]
MetricName = Literal["auroc", "auprc"]
METRIC_NAMES = ("auroc", "auprc", "brier_score", "ece")
TABLE_METADATA_COLUMNS = frozenset(
    {"video_id", "window_start_s", "window_end_s", "label"}
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    distillation: bool
    multiscale: bool


BASE_METHODS = (
    MethodSpec("current_logistic_baseline", False, False),
    MethodSpec("current_random_forest", False, False),
    MethodSpec("ms_stf_only", False, True),
    MethodSpec("teacher_only", False, True),
    MethodSpec("student_hard", False, True),
    MethodSpec("d_ms_stf", True, True),
)


@dataclass(frozen=True)
class FoldResult:
    protocol: Protocol
    fold: int
    method: str
    auroc: float | None
    auprc: float | None
    brier_score: float | None
    ece: float | None
    n_test: int
    n_positive: int
    n_negative: int
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class MetricAggregate:
    mean: float | None
    std: float | None
    n_available: int
    n_unavailable: int


@dataclass(frozen=True)
class AggregateResult:
    protocol: Protocol
    method: str
    distillation: bool
    multiscale: bool
    n_folds: int
    auroc: MetricAggregate
    auprc: MetricAggregate
    brier_score: MetricAggregate
    ece: MetricAggregate


@dataclass(frozen=True)
class DistillationFoldAudit:
    protocol: Protocol
    fold: int
    method: str
    outer_train_groups: tuple[str, ...]
    outer_test_groups: tuple[str, ...]
    inner_folds: tuple[InnerFoldAudit, ...]


@dataclass(frozen=True)
class PermutationResult:
    protocol: Protocol
    method: str
    metric: MetricName
    observed_score: float
    permutation_scores: tuple[float, ...]
    p_value: float
    n_permutations: int


@dataclass(frozen=True)
class EvaluationResult:
    feature_names: tuple[str, ...]
    multiscale_feature_names: tuple[str, ...]
    folds: tuple[FoldResult, ...]
    aggregates: tuple[AggregateResult, ...]
    audits: tuple[DistillationFoldAudit, ...]
    permutation: PermutationResult | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, stable output schema."""

        return {
            "schema_version": 1,
            "method": "D-MS-STF",
            "feature_names": list(self.feature_names),
            "multiscale_feature_names": list(self.multiscale_feature_names),
            "folds": [asdict(row) for row in self.folds],
            "aggregates": [asdict(row) for row in self.aggregates],
            "leakage_audits": [asdict(row) for row in self.audits],
            "permutation": (
                None if self.permutation is None else asdict(self.permutation)
            ),
        }


def evaluate_distilled_ms_stf(
    X: Any,
    y: Any,
    groups: Any,
    feature_names: Sequence[str],
    *,
    group_splits: int = 3,
    protocols: Iterable[Protocol] = ("group_kfold", "loso"),
    config: DistillationConfig = DistillationConfig(),
    multiscale_prefix: str = "ms_",
    include_calibrated_teacher: bool = False,
    n_permutations: int = 0,
    permutation_metric: MetricName = "auroc",
) -> EvaluationResult:
    """Run Stage F ablations with group-disjoint, fold-local fitting only."""

    features, labels, group_array, names = _validate_dataset(
        X, y, groups, feature_names
    )
    protocol_names = _validate_protocols(protocols)
    if not multiscale_prefix:
        raise ValueError("multiscale_prefix must not be empty")
    multiscale_mask = np.asarray(
        [name.startswith(multiscale_prefix) for name in names], dtype=bool
    )
    if not np.any(multiscale_mask):
        raise ValueError(
            f"no feature names start with multiscale prefix {multiscale_prefix!r}"
        )
    if np.all(multiscale_mask):
        raise ValueError("at least one non-multiscale baseline feature is required")

    methods = list(BASE_METHODS)
    if include_calibrated_teacher:
        methods.extend(
            (
                MethodSpec("teacher_calibrated", False, True),
                MethodSpec("d_ms_stf_calibrated", True, True),
            )
        )

    folds: list[FoldResult] = []
    audits: list[DistillationFoldAudit] = []
    for protocol in protocol_names:
        splits = _protocol_splits(protocol, group_array, group_splits)
        protocol_folds, protocol_audits = _evaluate_splits(
            features,
            labels,
            group_array,
            multiscale_mask,
            splits,
            protocol=protocol,
            config=config,
            methods=tuple(methods),
        )
        folds.extend(protocol_folds)
        audits.extend(protocol_audits)

    permutation = None
    if n_permutations:
        permutation = _permutation_test(
            features,
            labels,
            group_array,
            config=replace(config, calibrate_teacher=False),
            n_splits=group_splits,
            n_permutations=n_permutations,
            metric=permutation_metric,
        )

    return EvaluationResult(
        feature_names=names,
        multiscale_feature_names=tuple(np.asarray(names)[multiscale_mask].tolist()),
        folds=tuple(folds),
        aggregates=_aggregate_results(folds, methods),
        audits=tuple(audits),
        permutation=permutation,
    )


def write_evaluation_outputs(
    result: EvaluationResult, output_dir: Path
) -> tuple[Path, ...]:
    """Write aggregate numeric CSV, fold CSV, and a JSON audit report."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fold_path = destination / "fold_metrics.csv"
    aggregate_path = destination / "aggregate_metrics.csv"
    report_path = destination / "report.json"

    _write_fold_csv(result.folds, fold_path)
    _write_aggregate_csv(result, aggregate_path)
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fold_path, aggregate_path, report_path


def read_numeric_feature_table(
    path: Path,
    *,
    group_column: str = "video_id",
    label_column: str = "label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read a finite numeric feature CSV without accepting media-bearing fields."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        if len(columns) != len(set(columns)):
            raise ValueError("feature table columns must be unique")
        for required in (group_column, label_column):
            if required not in columns:
                raise ValueError(
                    f"feature table is missing required column {required!r}"
                )
        excluded = TABLE_METADATA_COLUMNS | {group_column, label_column}
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
                raw_label = row[label_column]
                label = int(raw_label)
                if str(label) != raw_label or label not in (0, 1):
                    raise ValueError("label must be exactly 0 or 1")
                numeric_row = [float(row[name]) for name in feature_names]
                if not all(isfinite(value) for value in numeric_row):
                    raise ValueError("features must be finite")
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid feature table row {row_number}: {exc}"
                ) from exc
            features.append(numeric_row)
            labels.append(label)
            groups.append(group)
    if not features:
        raise ValueError("feature table must contain at least one data row")
    return (
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(groups, dtype=object),
        feature_names,
    )


def _evaluate_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    multiscale_mask: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    protocol: Protocol,
    config: DistillationConfig,
    methods: tuple[MethodSpec, ...],
) -> tuple[list[FoldResult], list[DistillationFoldAudit]]:
    rows: list[FoldResult] = []
    audits: list[DistillationFoldAudit] = []
    base_config = replace(config, calibrate_teacher=False)
    calibrated_config = replace(config, calibrate_teacher=True)

    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        train_groups = tuple(sorted(set(map(str, groups[train_indices]))))
        test_groups = tuple(sorted(set(map(str, groups[test_indices]))))
        if set(train_groups) & set(test_groups):
            raise RuntimeError("outer split contains overlapping groups")

        if np.unique(y[train_indices]).size < 2:
            for method in methods:
                rows.append(
                    _fold_result(
                        protocol,
                        fold,
                        method.name,
                        y[test_indices],
                        test_groups,
                        scores=None,
                    )
                )
            continue

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        groups_train = groups[train_indices]
        baseline_train = X_train[:, ~multiscale_mask]
        baseline_test = X_test[:, ~multiscale_mask]

        baseline = train_logistic_regression_baseline(
            baseline_train, y_train, random_state=config.random_state + fold
        )
        rows.append(
            _fold_result(
                protocol,
                fold,
                "current_logistic_baseline",
                y[test_indices],
                test_groups,
                predict_scores(baseline, baseline_test),
            )
        )

        random_forest = train_random_forest_baseline(
            baseline_train, y_train, random_state=config.random_state + fold
        )
        rows.append(
            _fold_result(
                protocol,
                fold,
                "current_random_forest",
                y[test_indices],
                test_groups,
                predict_scores(random_forest, baseline_test),
            )
        )

        ms_model = train_logistic_regression_baseline(
            X_train, y_train, random_state=config.random_state + fold
        )
        rows.append(
            _fold_result(
                protocol,
                fold,
                "ms_stf_only",
                y[test_indices],
                test_groups,
                predict_scores(ms_model, X_test),
            )
        )

        teacher = train_teacher(
            X_train, y_train, groups=groups_train, config=base_config
        )
        rows.append(
            _fold_result(
                protocol,
                fold,
                "teacher_only",
                y[test_indices],
                test_groups,
                predict_scores(teacher, X_test),
            )
        )

        hard_student = train_hard_label_student(
            X_train, y_train, random_state=config.random_state + fold
        )
        rows.append(
            _fold_result(
                protocol,
                fold,
                "student_hard",
                y[test_indices],
                test_groups,
                predict_scores(hard_student, X_test),
            )
        )

        _append_distilled_result(
            rows,
            audits,
            X_train,
            X_test,
            y_train,
            y[test_indices],
            groups_train,
            train_groups,
            test_groups,
            protocol=protocol,
            fold=fold,
            method="d_ms_stf",
            config=base_config,
        )

        if any(method.name == "teacher_calibrated" for method in methods):
            calibrated_teacher = train_teacher(
                X_train,
                y_train,
                groups=groups_train,
                config=calibrated_config,
            )
            rows.append(
                _fold_result(
                    protocol,
                    fold,
                    "teacher_calibrated",
                    y[test_indices],
                    test_groups,
                    predict_scores(calibrated_teacher, X_test),
                )
            )
            _append_distilled_result(
                rows,
                audits,
                X_train,
                X_test,
                y_train,
                y[test_indices],
                groups_train,
                train_groups,
                test_groups,
                protocol=protocol,
                fold=fold,
                method="d_ms_stf_calibrated",
                config=calibrated_config,
            )
    return rows, audits


def _append_distilled_result(
    rows: list[FoldResult],
    audits: list[DistillationFoldAudit],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    train_groups: tuple[str, ...],
    test_groups: tuple[str, ...],
    *,
    protocol: Protocol,
    fold: int,
    method: str,
    config: DistillationConfig,
) -> None:
    soft_labels = cross_fitted_teacher_soft_labels(
        X_train, y_train, groups_train, config=config
    )
    for inner in soft_labels.folds:
        if set(test_groups) & (set(inner.train_groups) | set(inner.validation_groups)):
            raise RuntimeError("outer-test group reached inner distillation fitting")
    student = train_distilled_student(
        X_train, y_train, soft_labels.probabilities, config=config
    )
    rows.append(
        _fold_result(
            protocol,
            fold,
            method,
            y_test,
            test_groups,
            predict_scores(student, X_test),
        )
    )
    audits.append(
        DistillationFoldAudit(
            protocol=protocol,
            fold=fold,
            method=method,
            outer_train_groups=train_groups,
            outer_test_groups=test_groups,
            inner_folds=soft_labels.folds,
        )
    )


def _fold_result(
    protocol: Protocol,
    fold: int,
    method: str,
    y_test: np.ndarray,
    test_groups: tuple[str, ...],
    scores: np.ndarray | None,
) -> FoldResult:
    metrics = (
        {name: None for name in METRIC_NAMES}
        if scores is None
        else binary_classification_metrics(y_test, scores)
    )
    return FoldResult(
        protocol=protocol,
        fold=fold,
        method=method,
        auroc=metrics["auroc"],
        auprc=metrics["auprc"],
        brier_score=metrics["brier_score"],
        ece=metrics["ece"],
        n_test=int(y_test.size),
        n_positive=int(y_test.sum()),
        n_negative=int((y_test == 0).sum()),
        test_groups=test_groups,
    )


def _aggregate_results(
    folds: Sequence[FoldResult], methods: Sequence[MethodSpec]
) -> tuple[AggregateResult, ...]:
    method_lookup = {method.name: method for method in methods}
    results = []
    keys = sorted(set((row.protocol, row.method) for row in folds))
    for protocol, method_name in keys:
        selected = [
            row
            for row in folds
            if row.protocol == protocol and row.method == method_name
        ]
        spec = method_lookup[method_name]
        metrics = {
            name: _aggregate_metric(getattr(row, name) for row in selected)
            for name in METRIC_NAMES
        }
        results.append(
            AggregateResult(
                protocol=protocol,
                method=method_name,
                distillation=spec.distillation,
                multiscale=spec.multiscale,
                n_folds=len(selected),
                **metrics,
            )
        )
    return tuple(results)


def _aggregate_metric(values: Iterable[float | None]) -> MetricAggregate:
    collected = tuple(values)
    available = np.asarray([value for value in collected if value is not None])
    return MetricAggregate(
        mean=float(np.mean(available)) if available.size else None,
        std=float(np.std(available)) if available.size else None,
        n_available=int(available.size),
        n_unavailable=len(collected) - int(available.size),
    )


def _permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    config: DistillationConfig,
    n_splits: int,
    n_permutations: int,
    metric: MetricName,
) -> PermutationResult:
    if (
        isinstance(n_permutations, bool)
        or not isinstance(n_permutations, Integral)
        or n_permutations < 1
    ):
        raise ValueError("n_permutations must be a positive integer")
    if metric not in ("auroc", "auprc"):
        raise ValueError("permutation_metric must be 'auroc' or 'auprc'")
    splits = group_kfold_indices(groups, n_splits)
    observed = _distilled_mean_score(X, y, groups, splits, config, metric)
    rng = np.random.default_rng(config.random_state)
    null_scores = []
    for _ in range(n_permutations):
        permuted = shuffle_labels_within_groups(y, groups, random_state=rng)
        null_scores.append(
            _distilled_mean_score(X, permuted, groups, splits, config, metric)
        )
    extreme = sum(score >= observed for score in null_scores)
    return PermutationResult(
        protocol="group_kfold",
        method="d_ms_stf",
        metric=metric,
        observed_score=observed,
        permutation_scores=tuple(null_scores),
        p_value=float((1 + extreme) / (1 + n_permutations)),
        n_permutations=n_permutations,
    )


def _distilled_mean_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    config: DistillationConfig,
    metric: MetricName,
) -> float:
    scores = []
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        if np.unique(y[train_indices]).size < 2:
            continue
        fold_config = replace(config, random_state=config.random_state + fold)
        soft = cross_fitted_teacher_soft_labels(
            X[train_indices],
            y[train_indices],
            groups[train_indices],
            config=fold_config,
        )
        student = train_distilled_student(
            X[train_indices],
            y[train_indices],
            soft.probabilities,
            config=fold_config,
        )
        value = binary_classification_metrics(
            y[test_indices], predict_scores(student, X[test_indices])
        )[metric]
        if value is not None:
            scores.append(value)
    if not scores:
        raise ValueError(f"{metric} is unavailable on every permutation fold")
    return float(np.mean(scores))


def _protocol_splits(
    protocol: Protocol, groups: np.ndarray, group_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol == "group_kfold":
        return group_kfold_indices(groups, group_splits)
    return leave_one_group_out_indices(groups)


def _validate_protocols(protocols: Iterable[Protocol]) -> tuple[Protocol, ...]:
    values = tuple(protocols)
    if not values:
        raise ValueError("protocols must not be empty")
    if len(values) != len(set(values)) or any(
        value not in ("group_kfold", "loso") for value in values
    ):
        raise ValueError("protocols must contain unique 'group_kfold' and/or 'loso'")
    return values


def _validate_dataset(
    X: Any, y: Any, groups: Any, feature_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    features = np.asarray(X, dtype=float)
    raw_labels = np.asarray(y)
    labels = np.asarray(y, dtype=int)
    group_array = np.asarray(groups, dtype=object)
    names = tuple(feature_names)
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("X must be a finite two-dimensional feature matrix")
    if labels.ndim != 1 or labels.size != features.shape[0]:
        raise ValueError("y must be one-dimensional and match X rows")
    if not np.array_equal(labels, raw_labels) or not set(labels.tolist()) <= {0, 1}:
        raise ValueError("y must contain binary labels")
    if group_array.ndim != 1 or group_array.size != labels.size:
        raise ValueError("groups must be one-dimensional and match X rows")
    if any(value is None or not str(value).strip() for value in group_array):
        raise ValueError("groups must contain non-empty values")
    if len(names) != features.shape[1] or len(names) != len(set(names)):
        raise ValueError("feature_names must be unique and match X columns")
    if any(not name for name in names):
        raise ValueError("feature names must not be empty")
    return features, labels, group_array, names


def _write_fold_csv(rows: Sequence[FoldResult], path: Path) -> None:
    fieldnames = (
        "protocol",
        "fold",
        "method",
        *METRIC_NAMES,
        "n_test",
        "n_positive",
        "n_negative",
        "test_groups",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record["test_groups"] = ";".join(row.test_groups)
            writer.writerow(record)


def _write_aggregate_csv(result: EvaluationResult, path: Path) -> None:
    fields = ["protocol", "method", "distillation", "multiscale", "n_folds"]
    for name in METRIC_NAMES:
        fields.extend(
            (
                f"{name}_mean",
                f"{name}_std",
                f"{name}_n_available",
                f"{name}_n_unavailable",
            )
        )
    fields.append("permutation_p_value")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result.aggregates:
            record: dict[str, Any] = {
                "protocol": row.protocol,
                "method": row.method,
                "distillation": row.distillation,
                "multiscale": row.multiscale,
                "n_folds": row.n_folds,
            }
            for name in METRIC_NAMES:
                metric = getattr(row, name)
                record.update(
                    {
                        f"{name}_mean": metric.mean,
                        f"{name}_std": metric.std,
                        f"{name}_n_available": metric.n_available,
                        f"{name}_n_unavailable": metric.n_unavailable,
                    }
                )
            record["permutation_p_value"] = (
                result.permutation.p_value
                if result.permutation is not None
                and row.protocol == result.permutation.protocol
                and row.method == result.permutation.method
                else None
            )
            writer.writerow(record)
