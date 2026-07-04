import pytest

from ssbd_behavior.evaluation import (
    binary_classification_metrics,
    expected_calibration_error,
)


def test_binary_metrics_match_expected_values() -> None:
    metrics = binary_classification_metrics(
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], n_bins=2
    )

    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5 / 6)
    assert metrics["brier_score"] == pytest.approx(0.158125)
    assert metrics["ece"] == pytest.approx(0.0875)


def test_single_class_discrimination_metrics_are_explicitly_unavailable() -> None:
    metrics = binary_classification_metrics([0, 0, 0], [0.1, 0.2, 0.3])

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert metrics["brier_score"] == pytest.approx((0.01 + 0.04 + 0.09) / 3)
    assert metrics["ece"] is not None


def test_ece_handles_probability_one_in_final_bin() -> None:
    assert expected_calibration_error([0, 1], [0.0, 1.0], n_bins=5) == 0.0
