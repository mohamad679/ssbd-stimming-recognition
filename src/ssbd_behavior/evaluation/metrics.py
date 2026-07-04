"""Discrimination and calibration metrics for binary predictions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def binary_classification_metrics(
    y_true: Iterable[int], y_score: Iterable[float], *, n_bins: int = 10
) -> dict[str, float | None]:
    """Compute AUROC, AUPRC, Brier score, and fixed-bin ECE.

    AUROC and AUPRC are returned as ``None`` when ``y_true`` contains only one
    class because discrimination is undefined for such a fold.
    """

    labels, scores = _validate_inputs(y_true, y_score)
    return {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "brier_score": brier_score(labels, scores),
        "ece": expected_calibration_error(labels, scores, n_bins=n_bins),
    }


compute_binary_classification_metrics = binary_classification_metrics


def auroc(y_true: Iterable[int], y_score: Iterable[float]) -> float | None:
    """Return AUROC, or ``None`` when only one outcome class is present."""

    labels, scores = _validate_inputs(y_true, y_score)
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def auprc(y_true: Iterable[int], y_score: Iterable[float]) -> float | None:
    """Return AUPRC, or ``None`` when only one outcome class is present."""

    labels, scores = _validate_inputs(y_true, y_score)
    if np.unique(labels).size < 2:
        return None
    return float(average_precision_score(labels, scores))


def brier_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Return mean squared error between binary outcomes and probabilities."""

    labels, scores = _validate_inputs(y_true, y_score)
    return float(brier_score_loss(labels, scores))


def expected_calibration_error(
    y_true: Iterable[int], y_score: Iterable[float], *, n_bins: int = 10
) -> float:
    """Compute fixed-width expected calibration error on probability bins."""

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    labels, scores = _validate_inputs(y_true, y_score)
    bin_indices = np.minimum((scores * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        if not np.any(in_bin):
            continue
        calibration_gap = abs(float(scores[in_bin].mean() - labels[in_bin].mean()))
        ece += float(in_bin.mean()) * calibration_gap
    return ece


def _validate_inputs(
    y_true: Iterable[int], y_score: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    try:
        labels = np.asarray(list(y_true))
        scores = np.asarray(list(y_score), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and y_score must be one-dimensional iterables") from exc
    if labels.ndim != 1 or scores.ndim != 1 or labels.size == 0:
        raise ValueError("y_true and y_score must be non-empty one-dimensional arrays")
    if labels.size != scores.size:
        raise ValueError("y_true and y_score must have the same length")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("y_score must contain finite probabilities in [0, 1]")
    return labels.astype(int), scores
