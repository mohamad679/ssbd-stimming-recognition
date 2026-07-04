"""Leakage-resistant cross-validation splits grouped by video or child."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


def group_kfold_indices(
    groups: Iterable[Any], n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic GroupKFold indices after validating group labels."""

    group_array = _validate_groups(groups)
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    unique_group_count = len(set(group_array.tolist()))
    if n_splits > unique_group_count:
        raise ValueError(
            "n_splits cannot exceed the number of distinct groups "
            f"({unique_group_count})"
        )

    placeholder = np.zeros(group_array.size)
    splits = list(
        GroupKFold(n_splits=int(n_splits)).split(placeholder, groups=group_array)
    )
    for train_indices, test_indices in splits:
        validate_group_disjoint_split(train_indices, test_indices, group_array)
    return splits


def validate_group_disjoint_split(
    train_indices: Iterable[int],
    test_indices: Iterable[int],
    groups: Iterable[Any],
) -> None:
    """Raise ``ValueError`` if indices are invalid or their groups overlap."""

    group_array = _validate_groups(groups)
    train = _validate_indices(train_indices, "train_indices", group_array.size)
    test = _validate_indices(test_indices, "test_indices", group_array.size)

    shared_indices = np.intersect1d(train, test)
    if shared_indices.size:
        raise ValueError("train_indices and test_indices must not overlap")

    train_groups = set(group_array[train].tolist())
    test_groups = set(group_array[test].tolist())
    overlap = train_groups & test_groups
    if overlap:
        rendered = ", ".join(sorted(map(repr, overlap)))
        raise ValueError(f"group leakage detected across train and test: {rendered}")


def _validate_groups(groups: Iterable[Any]) -> np.ndarray:
    if groups is None:
        raise ValueError("groups are required for group-disjoint validation")
    try:
        group_array = np.asarray(list(groups), dtype=object)
    except TypeError as exc:
        raise ValueError("groups must be a one-dimensional iterable") from exc
    if group_array.ndim != 1 or group_array.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional iterable")

    for position, group in enumerate(group_array):
        missing = group is None or (isinstance(group, str) and not group.strip())
        try:
            missing = missing or bool(np.isscalar(group) and np.isnan(group))
        except TypeError:
            pass
        try:
            hash(group)
        except TypeError as exc:
            raise ValueError(f"group at position {position} must be hashable") from exc
        if missing:
            raise ValueError(f"group at position {position} is missing")
    return group_array


def _validate_indices(
    indices: Iterable[int], name: str, sample_count: int
) -> np.ndarray:
    try:
        values = list(indices)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{name} must contain only integer indices")
    array = np.asarray(values, dtype=int)
    if np.any(array < 0) or np.any(array >= sample_count):
        raise ValueError(f"{name} contains an out-of-range index")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return array
