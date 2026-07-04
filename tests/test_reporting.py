from ssbd_behavior.evaluation import (
    FoldMetricRow,
    format_fold_metrics,
    format_metric_summary,
    summarize_fold_metrics,
)


def _rows() -> list[FoldMetricRow]:
    return [
        FoldMetricRow(2, "model_b", None, None, 0.30, 0.20, 3, 0, 3, ("g3",)),
        FoldMetricRow(1, "model_b", 0.80, 0.70, 0.10, 0.10, 4, 2, 2, ("g2", "g1")),
    ]


def test_summary_ignores_unavailable_discrimination_metrics() -> None:
    summary = summarize_fold_metrics(_rows())[0]

    assert summary.auroc.mean == 0.8
    assert summary.auroc.n_available == 1
    assert summary.auroc.n_unavailable == 1
    assert summary.auprc.mean == 0.7
    assert summary.auprc.n_unavailable == 1
    assert summary.brier_score.mean == 0.2
    assert summary.brier_score.n_unavailable == 0


def test_reporting_format_is_deterministic_and_explicit() -> None:
    rows = _rows()

    assert format_fold_metrics(rows) == format_fold_metrics(reversed(rows))
    fold_output = format_fold_metrics(rows)
    assert fold_output.index("g1,g2") < fold_output.index("g3")
    assert "unavailable" in fold_output

    summary_output = format_metric_summary(summarize_fold_metrics(rows))
    assert "auroc=0.800000+/-0.000000(available=1,unavailable=1)" in summary_output
    assert "auprc=0.700000+/-0.000000(available=1,unavailable=1)" in summary_output
