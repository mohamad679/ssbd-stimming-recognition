"""Leave-one-group-out validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .splits import _validate_groups, validate_group_disjoint_split


@dataclass(frozen=True)
class LOSOFoldStatus:
    """Class availability and test composition for one held-out group."""

    group: str
    train_has_two_classes: bool
    test_has_two_classes: bool
    n_train: int
    n_test: int
    n_test_positive: int
    n_test_negative: int


def leave_one_group_out_indices(
    groups: Iterable[Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return one deterministic group-disjoint split per distinct group."""

    group_array = _validate_groups(groups)
    ordered_groups = tuple(dict.fromkeys(group_array.tolist()))
    if len(ordered_groups) < 2:
        raise ValueError(
            "leave-one-group-out validation requires at least 2 distinct groups"
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for group in ordered_groups:
        test_indices = np.flatnonzero(group_array == group).astype(int)
        train_indices = np.flatnonzero(group_array != group).astype(int)
        validate_group_disjoint_split(train_indices, test_indices, group_array)
        splits.append((train_indices, test_indices))
    return splits


def validate_loso_fold(
    y_train: Iterable[int],
    y_test: Iterable[int],
    group_name: Any,
) -> LOSOFoldStatus:
    """Return explicit LOSO fold status without inventing unavailable metrics.

    ``train_has_two_classes=False`` means model fitting must be skipped.
    ``test_has_two_classes=False`` means AUROC and AUPRC are unavailable.
    """

    train_labels = _validate_binary_labels(y_train, "y_train")
    test_labels = _validate_binary_labels(y_test, "y_test")
    group = _validate_group_name(group_name)
    return LOSOFoldStatus(
        group=group,
        train_has_two_classes=np.unique(train_labels).size == 2,
        test_has_two_classes=np.unique(test_labels).size == 2,
        n_train=int(train_labels.size),
        n_test=int(test_labels.size),
        n_test_positive=int(test_labels.sum()),
        n_test_negative=int((test_labels == 0).sum()),
    )


def _validate_binary_labels(labels: Iterable[int], name: str) -> np.ndarray:
    try:
        array = np.asarray(list(labels))
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional iterable")
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return array.astype(int)


def _validate_group_name(group_name: Any) -> str:
    missing = group_name is None or (
        isinstance(group_name, str) and not group_name.strip()
    )
    try:
        missing = missing or bool(np.isscalar(group_name) and np.isnan(group_name))
    except TypeError:
        pass
    if missing:
        raise ValueError("group_name is missing")
    return str(group_name)
