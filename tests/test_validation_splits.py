import numpy as np
import pytest

from ssbd_behavior.validation import (
    group_kfold_indices,
    validate_group_disjoint_split,
)


def test_group_kfold_never_leaks_groups() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

    splits = group_kfold_indices(groups, n_splits=4)

    assert len(splits) == 4
    for train_indices, test_indices in splits:
        assert set(groups[train_indices]).isdisjoint(groups[test_indices])
        validate_group_disjoint_split(train_indices, test_indices, groups)


def test_leakage_validator_catches_group_overlap() -> None:
    groups = ["video-a", "video-a", "video-b", "video-c"]

    with pytest.raises(ValueError, match="group leakage"):
        validate_group_disjoint_split([0, 2], [1, 3], groups)


@pytest.mark.parametrize(
    ("groups", "n_splits", "message"),
    [
        (None, 2, "groups are required"),
        (["a", None, "b"], 2, "missing"),
        (["a", "b"], 1, "at least 2"),
        (["a", "b"], 3, "distinct groups"),
    ],
)
def test_invalid_group_split_inputs_are_clear(groups, n_splits, message) -> None:
    with pytest.raises(ValueError, match=message):
        group_kfold_indices(groups, n_splits)
