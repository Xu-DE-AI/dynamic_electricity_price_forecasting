import numpy as np
import pandas as pd
from power_price.config import OPTIONAL_COVARIATES

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct delivery-time features using only lagged prices and known covariates.

    Lags start at 24 hours, allowing all predictors for a next-day delivery curve to
    be assembled before delivery. Optional system inputs are valid only if published
    by the forecast issue time.
    """
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    ts = data["timestamp"]
    hour, dow, doy = ts.dt.hour, ts.dt.dayofweek, ts.dt.dayofyear
    data["hour"], data["day_of_week"], data["month"] = hour, dow, ts.dt.month
    data["is_weekend"] = (dow >= 5).astype(int)
    for name, value, period in [("hour", hour, 24), ("dow", dow, 7), ("doy", doy, 365.25)]:
        data[f"{name}_sin"] = np.sin(2*np.pi*value/period)
        data[f"{name}_cos"] = np.cos(2*np.pi*value/period)
    for lag in (24, 48, 72, 168, 336):
        data[f"price_lag_{lag}"] = data["price_eur_mwh"].shift(lag)
    data["price_mean_prev_24h"] = data["price_eur_mwh"].shift(24).rolling(24, min_periods=18).mean()
    data["price_std_prev_168h"] = data["price_eur_mwh"].shift(24).rolling(168, min_periods=96).std()
    data["price_min_prev_168h"] = data["price_eur_mwh"].shift(24).rolling(168, min_periods=96).min()
    data["price_max_prev_168h"] = data["price_eur_mwh"].shift(24).rolling(168, min_periods=96).max()
    features = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos", "price_lag_24", "price_lag_48",
        "price_lag_72", "price_lag_168", "price_lag_336", "price_mean_prev_24h",
        "price_std_prev_168h", "price_min_prev_168h", "price_max_prev_168h"]
    features.extend(c for c in OPTIONAL_COVARIATES if c in data.columns)
    return data.reindex(columns=features)
