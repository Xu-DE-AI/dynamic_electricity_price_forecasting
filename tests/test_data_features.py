import pandas as pd
import pytest
from power_price.data import generate_benchmark_data, validate_market_data
from power_price.features import build_features

def test_benchmark_timestamps_are_unique_and_ordered():
    data = generate_benchmark_data(1200, 7)
    assert data.timestamp.is_monotonic_increasing and data.timestamp.is_unique

def test_duplicate_timestamps_are_rejected():
    data = generate_benchmark_data(1200, 7)
    data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="unique"):
        validate_market_data(data)

def test_lag_24_uses_previous_day_price():
    data = generate_benchmark_data(1200, 7)
    x = build_features(data)
    assert pd.isna(x.loc[23, "price_lag_24"])
    assert x.loc[24, "price_lag_24"] == pytest.approx(data.loc[0, "price_eur_mwh"])
