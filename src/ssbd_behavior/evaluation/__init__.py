"""Metrics for binary behavior-recognition evaluation."""

from .metrics import (
    auprc,
    auroc,
    binary_classification_metrics,
    brier_score,
    compute_binary_classification_metrics,
    expected_calibration_error,
)
from .reporting import (
    FoldMetricRow,
    MetricSummary,
    ModelMetricSummary,
    format_fold_metrics,
    format_metric_summary,
    summarize_fold_metrics,
)

__all__ = [
    "auprc",
    "auroc",
    "binary_classification_metrics",
    "brier_score",
    "compute_binary_classification_metrics",
    "expected_calibration_error",
    "FoldMetricRow",
    "MetricSummary",
    "ModelMetricSummary",
    "format_fold_metrics",
    "format_metric_summary",
    "summarize_fold_metrics",
]
