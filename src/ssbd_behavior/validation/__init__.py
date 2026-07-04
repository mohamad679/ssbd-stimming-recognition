"""Group-disjoint validation helpers."""

from .loso import LOSOFoldStatus, leave_one_group_out_indices, validate_loso_fold
from .permutation import (
    PermutationTestResult,
    permutation_test_score,
    shuffle_labels_within_groups,
)
from .splits import (
    group_kfold_indices,
    validate_group_disjoint_split,
)

__all__ = [
    "LOSOFoldStatus",
    "PermutationTestResult",
    "group_kfold_indices",
    "leave_one_group_out_indices",
    "permutation_test_score",
    "shuffle_labels_within_groups",
    "validate_group_disjoint_split",
    "validate_loso_fold",
]
