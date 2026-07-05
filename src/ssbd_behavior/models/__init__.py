"""Deterministic baseline classifiers for numeric window features."""

from .baselines import (
    predict_scores,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from .distilled_ms_stf import (
    CrossFittedSoftLabels,
    DistillationConfig,
    InnerFoldAudit,
    cross_fitted_teacher_soft_labels,
    soften_probabilities,
    train_distilled_student,
    train_hard_label_student,
    train_teacher,
)

__all__ = [
    "predict_scores",
    "CrossFittedSoftLabels",
    "DistillationConfig",
    "InnerFoldAudit",
    "cross_fitted_teacher_soft_labels",
    "soften_probabilities",
    "train_distilled_student",
    "train_hard_label_student",
    "train_teacher",
    "train_logistic_regression_baseline",
    "train_random_forest_baseline",
]
