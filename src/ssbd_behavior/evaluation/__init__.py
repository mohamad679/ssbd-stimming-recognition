"""Metrics for binary behavior-recognition evaluation."""

from .metrics import (
    auprc,
    auroc,
    binary_classification_metrics,
    brier_score,
    compute_binary_classification_metrics,
    expected_calibration_error,
)
from .distilled_ms_stf_eval import (
    AggregateResult,
    DistillationFoldAudit,
    EvaluationResult,
    FoldResult,
    MetricAggregate,
    PermutationResult,
    evaluate_distilled_ms_stf,
    read_numeric_feature_table,
    write_evaluation_outputs,
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
from .svg_validation import validate_svg_file, validate_svg_files

__all__ = [
    "auprc",
    "auroc",
    "binary_classification_metrics",
    "brier_score",
    "compute_binary_classification_metrics",
    "expected_calibration_error",
    "AggregateResult",
    "DistillationFoldAudit",
    "EvaluationResult",
    "FoldResult",
    "MetricAggregate",
    "PermutationResult",
    "evaluate_distilled_ms_stf",
    "read_numeric_feature_table",
    "write_evaluation_outputs",
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
    "validate_svg_file",
    "validate_svg_files",
]
