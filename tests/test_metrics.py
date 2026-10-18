from power_price.metrics import interval_metrics, pinball_loss, point_metrics

def test_metrics_support_negative_and_zero_prices():
    result = point_metrics([-10, 0, 50], [-8, 2, 40])
    assert result["mae_eur_mwh"] >= 0 and result["rmse_eur_mwh"] >= 0
    assert result["smape_pct"] >= 0

def test_quantile_metrics():
    assert interval_metrics([1, 2, 3], [0, 1, 2], [2, 3, 4])["coverage_pct"] == 100
    assert pinball_loss([1, 2], [0, 1], .5) >= 0
