"""Deterministic, non-persistent reporting for grouped baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


METRIC_NAMES = ("auroc", "auprc", "brier_score", "ece")


@dataclass(frozen=True)
class FoldMetricRow:
    """Metrics and test-set composition for one model on one fold."""

    fold: int
    model_name: str
    auroc: float | None
    auprc: float | None
    brier_score: float | None
    ece: float | None
    n_test: int
    n_positive: int
    n_negative: int
    groups_tested: tuple[str, ...]


@dataclass(frozen=True)
class MetricSummary:
    """Availability and aggregate statistics for one metric."""

    mean: float | None
    std: float | None
    n_available: int
    n_unavailable: int


@dataclass(frozen=True)
class ModelMetricSummary:
    """Per-model aggregate across fold rows."""

    model_name: str
    n_folds: int
    auroc: MetricSummary
    auprc: MetricSummary
    brier_score: MetricSummary
    ece: MetricSummary


def summarize_fold_metrics(rows: Iterable[FoldMetricRow]) -> tuple[ModelMetricSummary, ...]:
    """Summarize rows per model without imputing unavailable metric values."""

    rows_by_model: dict[str, list[FoldMetricRow]] = {}
    for row in rows:
        rows_by_model.setdefault(row.model_name, []).append(row)

    summaries = []
    for model_name in sorted(rows_by_model):
        model_rows = rows_by_model[model_name]
        aggregates = {
            name: _summarize_values(getattr(row, name) for row in model_rows)
            for name in METRIC_NAMES
        }
        summaries.append(
            ModelMetricSummary(
                model_name=model_name,
                n_folds=len(model_rows),
                **aggregates,
            )
        )
    return tuple(summaries)


def format_fold_metrics(rows: Iterable[FoldMetricRow]) -> str:
    """Return a stable, human-readable fold table."""

    ordered_rows = sorted(rows, key=lambda row: (row.fold, row.model_name))
    header = (
        "fold model                 auroc      auprc      brier_score ece        "
        "n_test n_pos n_neg groups_tested"
    )
    lines = [header]
    for row in ordered_rows:
        groups = ",".join(sorted(row.groups_tested))
        lines.append(
            f"{row.fold:>4} {row.model_name:<21} "
            f"{_format_value(row.auroc):>10} {_format_value(row.auprc):>10} "
            f"{_format_value(row.brier_score):>11} {_format_value(row.ece):>10} "
            f"{row.n_test:>6} {row.n_positive:>5} {row.n_negative:>5} {groups}"
        )
    return "\n".join(lines)


def format_metric_summary(
    summary: ModelMetricSummary | Iterable[ModelMetricSummary],
) -> str:
    """Return stable summary lines for one model or an iterable of models."""

    summaries = (
        (summary,) if isinstance(summary, ModelMetricSummary) else tuple(summary)
    )
    lines = []
    for model_summary in sorted(summaries, key=lambda item: item.model_name):
        lines.append(
            f"model={model_summary.model_name} folds={model_summary.n_folds} "
            + " ".join(
                f"{name}={_format_aggregate(getattr(model_summary, name))}"
                for name in METRIC_NAMES
            )
        )
    return "\n".join(lines)


def _summarize_values(values: Iterable[float | None]) -> MetricSummary:
    collected = tuple(values)
    available = tuple(value for value in collected if value is not None)
    if not available:
        return MetricSummary(None, None, 0, len(collected))
    mean = sum(available) / len(available)
    variance = sum((value - mean) ** 2 for value in available) / len(available)
    return MetricSummary(mean, sqrt(variance), len(available), len(collected) - len(available))


def _format_value(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


def _format_aggregate(metric: MetricSummary) -> str:
    if metric.mean is None:
        statistics = "unavailable"
    else:
        statistics = f"{metric.mean:.6f}+/-{metric.std:.6f}"
    return (
        f"{statistics}(available={metric.n_available},"
        f"unavailable={metric.n_unavailable})"
    )
