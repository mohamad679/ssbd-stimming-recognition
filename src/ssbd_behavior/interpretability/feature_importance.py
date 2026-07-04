"""Model-native feature importance helpers for numeric feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureImportanceRecord:
    """One ranked, model-native exploratory feature importance."""

    feature: str
    importance: float
    source: str
    rank: int
    notes: str | None = None


def extract_model_feature_importance(
    model: Any,
    feature_names: Sequence[str],
    *,
    source: str,
) -> list[FeatureImportanceRecord]:
    """Extract deterministic, non-causal importances from a fitted sklearn model."""

    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must contain at least one feature")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("feature_names must be unique")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")

    estimator = _model_native_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if values.ndim != 1:
            raise ValueError("model feature_importances_ must be one-dimensional")
        notes = "model-native feature importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 1:
            values = np.abs(coefficients)
            notes = "absolute model coefficient"
        elif coefficients.ndim == 2 and coefficients.shape[0] > 0:
            values = np.mean(np.abs(coefficients), axis=0)
            notes = (
                "absolute model coefficient"
                if coefficients.shape[0] == 1
                else "mean absolute model coefficient across classes"
            )
        else:
            raise ValueError("model coef_ must be a non-empty one- or two-dimensional array")
    else:
        raise ValueError(
            "unsupported model: expected fitted feature_importances_ or coef_"
        )

    if len(values) != len(names):
        raise ValueError(
            "feature count mismatch: model exposes "
            f"{len(values)} importances for {len(names)} feature names"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("model feature importances must be finite")

    ordered = sorted(
        zip(names, values, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [
        FeatureImportanceRecord(
            feature=feature,
            importance=float(importance),
            source=source,
            rank=rank,
            notes=notes,
        )
        for rank, (feature, importance) in enumerate(ordered, start=1)
    ]


def summarize_top_features(
    records: Iterable[FeatureImportanceRecord], *, top_k: int = 10
) -> list[FeatureImportanceRecord]:
    """Return a deterministic top-k ranking without causal interpretation."""

    if not isinstance(top_k, Integral) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    ordered = sorted(
        records,
        key=lambda record: (-record.importance, record.feature, record.source),
    )
    return ordered[: int(top_k)]


def _model_native_estimator(model: Any) -> Any:
    """Return a model exposing native importance attributes, including pipelines."""

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        return model

    steps = getattr(model, "steps", None)
    if steps:
        final_estimator = steps[-1][1]
        if hasattr(final_estimator, "feature_importances_") or hasattr(
            final_estimator, "coef_"
        ):
            return final_estimator
    return model
