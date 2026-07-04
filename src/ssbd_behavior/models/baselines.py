"""Simple, deterministic binary-classification baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42


def train_logistic_regression_baseline(
    X: Any, y: Any, *, random_state: int = DEFAULT_RANDOM_STATE
) -> Pipeline:
    """Fit a scaled, class-weighted logistic-regression baseline."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return model.fit(X, y)


def train_random_forest_baseline(
    X: Any, y: Any, *, random_state: int = DEFAULT_RANDOM_STATE
) -> RandomForestClassifier:
    """Fit a small, class-weighted random-forest baseline."""

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    return model.fit(X, y)


def predict_scores(model: Any, X: Any) -> np.ndarray:
    """Return the fitted model's probability for the positive class (label 1)."""

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("model must be a fitted probabilistic classifier")

    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if positive_columns.size != 1:
        raise ValueError("model must have exactly one positive class labeled 1")

    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != classes.size:
        raise ValueError("model returned an invalid probability matrix")
    return probabilities[:, int(positive_columns[0])]
