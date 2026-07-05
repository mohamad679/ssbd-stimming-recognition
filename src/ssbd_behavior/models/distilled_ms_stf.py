"""Leakage-safe teacher/student models for D-MS-STF evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ssbd_behavior.models.baselines import predict_scores
from ssbd_behavior.validation.splits import group_kfold_indices


TeacherKind = Literal["extra_trees", "random_forest", "hist_gradient_boosting"]


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration shared by fold-local teacher and student fitting.

    ``alpha`` is the hard-label weight. The complementary weight is applied to
    temperature-softened, inner-cross-fitted teacher probabilities.
    """

    teacher_kind: TeacherKind = "extra_trees"
    n_estimators: int = 200
    inner_splits: int = 3
    alpha: float = 0.5
    temperature: float = 1.0
    calibrate_teacher: bool = False
    calibration_splits: int = 3
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.teacher_kind not in (
            "extra_trees",
            "random_forest",
            "hist_gradient_boosting",
        ):
            raise ValueError("unsupported teacher_kind")
        _positive_integer(self.n_estimators, "n_estimators")
        _at_least_two(self.inner_splits, "inner_splits")
        _at_least_two(self.calibration_splits, "calibration_splits")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, Real)
            or not isfinite(float(self.alpha))
            or not 0.0 <= self.alpha <= 1.0
        ):
            raise ValueError("alpha must be a finite number between 0 and 1")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be a positive finite number")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, Integral
        ):
            raise ValueError("random_state must be an integer")


@dataclass(frozen=True)
class InnerFoldAudit:
    """Group membership used for one cross-fitted soft-label partition."""

    fold: int
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class CrossFittedSoftLabels:
    """Teacher probabilities and their group-disjoint training audit."""

    probabilities: np.ndarray
    folds: tuple[InnerFoldAudit, ...]


def train_teacher(
    X: Any,
    y: Any,
    *,
    groups: Any | None = None,
    config: DistillationConfig | None = None,
) -> Any:
    """Fit one teacher using only the samples supplied by the caller."""

    config = config or DistillationConfig()
    features, labels = _validate_xy(X, y)
    estimator = _make_teacher(config)
    if config.calibrate_teacher:
        if groups is None:
            raise ValueError("groups are required for fold-local teacher calibration")
        group_array = _validate_groups(groups, labels.size)
        split_count = min(config.calibration_splits, len(set(group_array.tolist())))
        if split_count < 2:
            raise ValueError("teacher calibration requires at least two groups")
        calibration_cv = group_kfold_indices(group_array, split_count)
        _validate_training_splits(labels, calibration_cv, require_test_classes=True)
        estimator = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=calibration_cv,
        )
    return estimator.fit(features, labels)


def cross_fitted_teacher_soft_labels(
    X: Any,
    y: Any,
    groups: Any,
    *,
    config: DistillationConfig | None = None,
) -> CrossFittedSoftLabels:
    """Generate soft labels without fitting any sample's teacher on that sample."""

    config = config or DistillationConfig()
    features, labels = _validate_xy(X, y)
    group_array = _validate_groups(groups, labels.size)
    split_count = min(config.inner_splits, len(set(group_array.tolist())))
    if split_count < 2:
        raise ValueError("cross-fitted distillation requires at least two groups")
    splits = group_kfold_indices(group_array, split_count)
    _validate_training_splits(labels, splits)

    probabilities = np.full(labels.size, np.nan, dtype=float)
    audit_rows = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        fold_config = replace(config, random_state=config.random_state + fold)
        teacher = train_teacher(
            features[train_indices],
            labels[train_indices],
            groups=group_array[train_indices],
            config=fold_config,
        )
        probabilities[validation_indices] = predict_scores(
            teacher, features[validation_indices]
        )
        train_groups = tuple(sorted(set(map(str, group_array[train_indices]))))
        validation_groups = tuple(
            sorted(set(map(str, group_array[validation_indices])))
        )
        if set(train_groups) & set(validation_groups):
            raise RuntimeError("inner cross-fitting produced overlapping groups")
        audit_rows.append(
            InnerFoldAudit(
                fold=fold,
                train_groups=train_groups,
                validation_groups=validation_groups,
                train_indices=tuple(map(int, train_indices)),
                validation_indices=tuple(map(int, validation_indices)),
            )
        )

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("inner cross-fitting did not score every training sample")
    return CrossFittedSoftLabels(probabilities=probabilities, folds=tuple(audit_rows))


def train_hard_label_student(
    X: Any,
    y: Any,
    *,
    random_state: int = 42,
) -> Pipeline:
    """Fit the lightweight student against hard labels only."""

    features, labels = _validate_xy(X, y)
    return _fit_soft_target_logistic(
        features,
        labels,
        labels.astype(float),
        random_state=random_state,
    )


def train_distilled_student(
    X: Any,
    y: Any,
    teacher_probabilities: Any,
    *,
    config: DistillationConfig | None = None,
) -> Pipeline:
    """Fit a student to the configured hard/soft Bernoulli target mixture.

    Scikit-learn classifiers do not accept fractional class labels. Each sample
    is therefore duplicated once per binary target and weighted by ``1-q`` and
    ``q``. This is algebraically the same Bernoulli log-loss for soft target
    ``q`` and keeps the implementation within standard, tested estimators.
    """

    config = config or DistillationConfig()
    features, labels = _validate_xy(X, y)
    teacher_scores = _validate_probabilities(teacher_probabilities, labels.size)
    softened = soften_probabilities(teacher_scores, config.temperature)
    targets = config.alpha * labels + (1.0 - config.alpha) * softened
    return _fit_soft_target_logistic(
        features,
        labels,
        targets,
        random_state=config.random_state,
    )


def soften_probabilities(probabilities: Any, temperature: float) -> np.ndarray:
    """Apply binary logit temperature scaling without fitting any parameters."""

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be a positive finite number")
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("probabilities must be a finite one-dimensional array")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be between 0 and 1")
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped)) / float(temperature)
    return 1.0 / (1.0 + np.exp(-logits))


def _make_teacher(config: DistillationConfig) -> Any:
    if config.teacher_kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
            random_state=config.random_state,
        )
    if config.teacher_kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
            random_state=config.random_state,
        )
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=config.n_estimators,
        l2_regularization=1.0,
        random_state=config.random_state,
    )


def _fit_soft_target_logistic(
    features: np.ndarray,
    hard_labels: np.ndarray,
    soft_targets: np.ndarray,
    *,
    random_state: int,
) -> Pipeline:
    sample_balance = compute_sample_weight("balanced", hard_labels)
    duplicated_features = np.repeat(features, 2, axis=0)
    duplicated_labels = np.tile(np.asarray([0, 1], dtype=int), hard_labels.size)
    target_weights = np.column_stack((1.0 - soft_targets, soft_targets)).reshape(-1)
    duplicated_weights = np.repeat(sample_balance, 2) * target_weights
    positive_weight = duplicated_weights > 0.0

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=int(random_state),
                    solver="liblinear",
                ),
            ),
        ]
    )
    return model.fit(
        duplicated_features[positive_weight],
        duplicated_labels[positive_weight],
        classifier__sample_weight=duplicated_weights[positive_weight],
    )


def _validate_xy(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")
    if labels.ndim != 1 or labels.size != features.shape[0]:
        raise ValueError("y must be one-dimensional and match X rows")
    if not np.array_equal(labels, np.asarray(y)) or not set(labels.tolist()) <= {0, 1}:
        raise ValueError("y must contain only integer binary labels")
    if np.unique(labels).size < 2:
        raise ValueError("model fitting requires both binary classes")
    return features, labels


def _validate_probabilities(values: Any, expected_size: int) -> np.ndarray:
    probabilities = soften_probabilities(values, 1.0)
    if probabilities.size != expected_size:
        raise ValueError("teacher probabilities must match X rows")
    return probabilities


def _validate_groups(groups: Any, expected_size: int) -> np.ndarray:
    result = np.asarray(groups, dtype=object)
    if result.ndim != 1 or result.size != expected_size:
        raise ValueError("groups must be one-dimensional and match X rows")
    if any(value is None or not str(value).strip() for value in result):
        raise ValueError("groups must contain non-empty values")
    return result


def _validate_training_splits(
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    require_test_classes: bool = False,
) -> None:
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        if np.unique(labels[train_indices]).size < 2:
            raise ValueError(f"inner fold {fold} training set has only one class")
        if require_test_classes and np.unique(labels[test_indices]).size < 2:
            raise ValueError(
                f"calibration fold {fold} validation set has only one class"
            )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _at_least_two(value: int, name: str) -> int:
    result = _positive_integer(value, name)
    if result < 2:
        raise ValueError(f"{name} must be at least 2")
    return result
