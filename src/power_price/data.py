from __future__ import annotations
import numpy as np
import pandas as pd
import requests

REQUIRED_COLUMNS = {"timestamp", "price_eur_mwh"}
OPTIONAL_NUMERIC = ["load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw", "net_import_forecast_mw"]

def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized market data and preserve all distinct UTC delivery periods."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
        raise ValueError("timestamp must contain valid, unique delivery-period timestamps")
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["price_eur_mwh"] = pd.to_numeric(data["price_eur_mwh"], errors="coerce")
    if data["price_eur_mwh"].isna().any() or not np.isfinite(data["price_eur_mwh"]).all():
        raise ValueError("price_eur_mwh must contain finite numeric values")
    for column in OPTIONAL_NUMERIC:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
            if data[column].isna().any() or not np.isfinite(data[column]).all():
                raise ValueError(f"{column} must contain finite values when supplied")
    return data

def load_market_csv(path: str) -> pd.DataFrame:
    return validate_market_data(pd.read_csv(path))

def fetch_energy_charts_prices(start: str, end: str, zone: str = "DE-LU") -> pd.DataFrame:
    """Fetch public historical day-ahead prices; fail explicitly on schema changes."""
    response = requests.get("https://api.energy-charts.info/price",
        params={"bzn": zone, "start": start, "end": end}, timeout=(10, 60))
    response.raise_for_status()
    payload = response.json()
    if "unix_seconds" not in payload or "price" not in payload:
        raise ValueError("Energy-Charts response lacks unix_seconds or price")
    if len(payload["unix_seconds"]) != len(payload["price"]):
        raise ValueError("Energy-Charts timestamp and price arrays differ in length")
    return validate_market_data(pd.DataFrame({
        "timestamp": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True),
        "price_eur_mwh": payload["price"],
    }))

def generate_benchmark_data(rows: int = 26280, seed: int = 41) -> pd.DataFrame:
    """Create deterministic synthetic prices with seasonality, spikes and negative values."""
    if rows < 504:
        raise ValueError("At least 504 hourly rows are required")
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=rows, freq="h", tz="UTC")
    hour, dow, doy = ts.hour.to_numpy(), ts.dayofweek.to_numpy(), ts.dayofyear.to_numpy()
    daily = 18*np.sin(2*np.pi*(hour-8)/24)
    weekly = 7*np.cos(2*np.pi*dow/7)
    seasonal = 22*np.sin(2*np.pi*(doy-30)/365.25)
    trend = np.linspace(0, 8, rows)
    renewable = np.clip(.5+.3*np.sin(2*np.pi*(hour+4)/24)+rng.normal(0,.1,rows), 0, 1)
    noise = rng.standard_t(5, rows)*7
    spike_mask = rng.random(rows) < .012
    spikes = spike_mask*rng.choice([-1, 1], rows)*rng.uniform(80, 240, rows)
    price = 72+daily+weekly+seasonal+trend+14*np.sin(2*np.pi*(hour-17)/24)-35*(renewable-.5)+noise+spikes
    price = np.where(rng.random(rows)<.008, -rng.uniform(1,35,rows), price)
    return validate_market_data(pd.DataFrame({
        "timestamp": ts, "price_eur_mwh": price,
        "wind_forecast_mw": 18000+9000*renewable,
        "solar_forecast_mw": np.maximum(0,12000*np.sin(np.pi*(hour-6)/12)),
        "load_forecast_mw": 54000+7000*np.sin(2*np.pi*(hour-8)/24)+2500*(dow<5),
        "net_import_forecast_mw": rng.normal(0,2500,rows),
    }))
