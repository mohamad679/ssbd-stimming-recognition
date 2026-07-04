"""Within-group permutation testing for group-disjoint validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import Any

import numpy as np

from ..models import predict_scores
from .splits import _validate_groups, group_kfold_indices, validate_group_disjoint_split


Trainer = Callable[[Any, Any], Any]
Scorer = Callable[[Iterable[int], Iterable[float]], float | None]


@dataclass(frozen=True)
class PermutationTestResult:
    """Observed and null scores for a group-aware permutation test."""

    observed_score: float
    permutation_scores: tuple[float, ...]
    p_value: float
    n_permutations: int
    scoring_name: str
    model_name: str
    n_scored_folds: int
    n_unavailable_folds: int
    model_config_note: str | None = None


@dataclass(frozen=True)
class _AggregatedFoldScore:
    mean_score: float | None
    n_scored_folds: int
    n_unavailable_folds: int


def shuffle_labels_within_groups(
    y: Iterable[Any],
    groups: Iterable[Any],
    *,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Shuffle labels independently within each group."""

    labels = _validate_label_array(y, "y")
    group_array = _validate_groups(groups)
    if labels.size != group_array.size:
        raise ValueError("y and groups must have the same length")

    rng = _coerce_random_generator(random_state)
    shuffled = labels.copy()
    for group in dict.fromkeys(group_array.tolist()):
        group_indices = np.flatnonzero(group_array == group)
        shuffled[group_indices] = labels[group_indices][rng.permutation(group_indices.size)]
    return shuffled


def permutation_test_score(
    X: Any,
    y: Iterable[int],
    groups: Iterable[Any],
    *,
    model_trainer: Trainer,
    scorer: Scorer,
    n_permutations: int,
    random_state: int | np.random.Generator | None,
    splits: Sequence[tuple[Iterable[int], Iterable[int]]] | None = None,
    n_splits: int = 3,
) -> PermutationTestResult:
    """Run a within-group permutation test using group-disjoint validation.

    The supplied ``model_trainer`` is called directly for every observed and
    permuted fit so the permutation run reuses the benchmark trainer without
    permutation-specific configuration overrides.

    The supplied ``scorer`` must be a larger-is-better metric because the
    p-value is computed as ``(1 + count(null >= observed)) / (1 + n_permutations)``.
    """

    features = _validate_feature_matrix(X)
    labels = _validate_binary_labels(y, "y")
    group_array = _validate_groups(groups)
    if features.shape[0] != labels.size or labels.size != group_array.size:
        raise ValueError("X, y, and groups must describe the same number of rows")

    permutation_count = _validate_positive_integer(
        n_permutations, name="n_permutations"
    )
    split_indices = _resolve_splits(
        groups=group_array,
        splits=splits,
        n_splits=n_splits,
    )

    observed = _aggregate_fold_scores(
        X=features,
        y=labels,
        splits=split_indices,
        model_trainer=model_trainer,
        scorer=scorer,
    )
    scoring_name, _ = _describe_callable(scorer)
    if observed.mean_score is None:
        raise ValueError(
            f"scoring {scoring_name!r} is unavailable on every validation fold"
        )

    rng = _coerce_random_generator(random_state)
    permutation_scores: list[float] = []
    for _ in range(permutation_count):
        permuted_labels = shuffle_labels_within_groups(
            labels,
            group_array,
            random_state=rng,
        )
        aggregated = _aggregate_fold_scores(
            X=features,
            y=permuted_labels,
            splits=split_indices,
            model_trainer=model_trainer,
            scorer=scorer,
        )
        if aggregated.mean_score is None:
            raise RuntimeError(
                "permutation produced no available fold scores; label shuffling "
                "should preserve fold metric availability"
            )
        permutation_scores.append(aggregated.mean_score)

    observed_score = observed.mean_score
    assert observed_score is not None
    extreme_count = sum(score >= observed_score for score in permutation_scores)
    p_value = (1 + extreme_count) / (1 + permutation_count)
    model_name, model_config_note = _describe_callable(model_trainer)
    return PermutationTestResult(
        observed_score=observed_score,
        permutation_scores=tuple(permutation_scores),
        p_value=float(p_value),
        n_permutations=permutation_count,
        scoring_name=scoring_name,
        model_name=model_name,
        n_scored_folds=observed.n_scored_folds,
        n_unavailable_folds=observed.n_unavailable_folds,
        model_config_note=model_config_note,
    )


def _aggregate_fold_scores(
    *,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    model_trainer: Trainer,
    scorer: Scorer,
) -> _AggregatedFoldScore:
    fold_scores: list[float] = []
    unavailable_fold_count = 0
    for train_indices, test_indices in splits:
        if np.unique(y[train_indices]).size < 2:
            unavailable_fold_count += 1
            continue

        model = model_trainer(X[train_indices], y[train_indices])
        y_score = predict_scores(model, X[test_indices])
        fold_score = scorer(y[test_indices], y_score)
        if fold_score is None:
            unavailable_fold_count += 1
            continue

        numeric_score = float(fold_score)
        if not np.isfinite(numeric_score):
            raise ValueError("scorer must return finite numeric values or None")
        fold_scores.append(numeric_score)

    if not fold_scores:
        return _AggregatedFoldScore(
            mean_score=None,
            n_scored_folds=0,
            n_unavailable_folds=unavailable_fold_count,
        )
    return _AggregatedFoldScore(
        mean_score=float(sum(fold_scores) / len(fold_scores)),
        n_scored_folds=len(fold_scores),
        n_unavailable_folds=unavailable_fold_count,
    )


def _resolve_splits(
    *,
    groups: np.ndarray,
    splits: Sequence[tuple[Iterable[int], Iterable[int]]] | None,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if splits is None:
        return group_kfold_indices(groups, n_splits=n_splits)

    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    for split_number, (train_indices, test_indices) in enumerate(splits, start=1):
        train_array = np.asarray(list(train_indices), dtype=int)
        test_array = np.asarray(list(test_indices), dtype=int)
        try:
            validate_group_disjoint_split(train_array, test_array, groups)
        except ValueError as exc:
            raise ValueError(f"invalid split {split_number}: {exc}") from exc
        normalized.append((train_array, test_array))
    if not normalized:
        raise ValueError("splits must contain at least one train/test split")
    return normalized


def _validate_feature_matrix(X: Any) -> np.ndarray:
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a finite two-dimensional numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional numeric matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite numeric values")
    return array


def _validate_binary_labels(labels: Iterable[int], name: str) -> np.ndarray:
    array = _validate_label_array(labels, name)
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return array.astype(int)


def _validate_label_array(labels: Iterable[Any], name: str) -> np.ndarray:
    try:
        array = np.asarray(list(labels))
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional iterable")
    return array


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


def _coerce_random_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _describe_callable(callable_obj: Any) -> tuple[str, str | None]:
    if isinstance(callable_obj, partial):
        base_name, _ = _describe_callable(callable_obj.func)
        notes = []
        if callable_obj.args:
            notes.append(f"partial_args={len(callable_obj.args)}")
        if callable_obj.keywords:
            rendered = ", ".join(
                f"{key}={value!r}" for key, value in sorted(callable_obj.keywords.items())
            )
            notes.append(f"partial_kwargs={rendered}")
        return base_name, "; ".join(notes) or None

    name = getattr(callable_obj, "__name__", callable_obj.__class__.__name__)
    return str(name), None


__all__ = [
    "PermutationTestResult",
    "permutation_test_score",
    "shuffle_labels_within_groups",
]
