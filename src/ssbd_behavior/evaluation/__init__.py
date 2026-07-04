"""Metrics for binary behavior-recognition evaluation."""

from .metrics import (
    auprc,
    auroc,
    binary_classification_metrics,
    brier_score,
    compute_binary_classification_metrics,
    expected_calibration_error,
)
from .provenance import (
    ArtifactRecord,
    build_artifact_records,
    read_artifact_manifest,
    sha256_file,
    write_artifact_manifest,
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
    "ArtifactRecord",
    "build_artifact_records",
    "read_artifact_manifest",
    "sha256_file",
    "write_artifact_manifest",
    "FoldMetricRow",
    "MetricSummary",
    "ModelMetricSummary",
    "format_fold_metrics",
    "format_metric_summary",
    "summarize_fold_metrics",
]
