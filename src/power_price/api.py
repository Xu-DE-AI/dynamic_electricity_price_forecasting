import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from power_price.features import build_features

app = FastAPI(title="Electricity Price Forecast API", version="1.0.0",
    description="Probabilistic 24-hour day-ahead wholesale electricity price forecasts.")
MODEL_PATH = Path(os.getenv("POWER_PRICE_MODEL_PATH", "artifacts/model_bundle.joblib"))

class HistoryRecord(BaseModel):
    timestamp: str
    price_eur_mwh: float
    load_forecast_mw: float | None = None
    wind_forecast_mw: float | None = None
    solar_forecast_mw: float | None = None
    net_import_forecast_mw: float | None = None

class ForecastRequest(BaseModel):
    history: list[HistoryRecord] = Field(min_length=504, description="At least 21 days of hourly history")
    horizon_hours: int = Field(default=24, ge=1, le=24)

def _load_bundle():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact is not available at {MODEL_PATH}")
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded") from exc
    if "models" not in bundle or "feature_columns" not in bundle:
        raise HTTPException(status_code=500, detail="Model artifact schema is invalid")
    return bundle

@app.get("/health")
def health():
    return {"status": "ok", "model_artifact_present": MODEL_PATH.exists()}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    bundle = _load_bundle()
    frame = pd.DataFrame([row.model_dump() for row in request.history])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise HTTPException(status_code=422, detail="History timestamps must be valid and unique")
    # Reject unordered requests instead of silently changing the caller's as-of history.
    if not frame["timestamp"].is_monotonic_increasing:
        raise HTTPException(status_code=422, detail="History must be supplied chronologically")
    if (frame["timestamp"].dt.minute != 0).any() or (frame["timestamp"].dt.second != 0).any():
        raise HTTPException(status_code=422, detail="History must use hourly delivery timestamps")
    latest = frame["timestamp"].max()
    future_times = pd.date_range(latest + pd.Timedelta(hours=1), periods=request.horizon_hours, freq="h")
    # No future target values are inserted. Since all target lags are >=24h, each requested
    # hour's price features remain derived from observed history rather than predictions.
    future = pd.DataFrame({"timestamp": future_times, "price_eur_mwh": np.nan})
    for col in ("load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw", "net_import_forecast_mw"):
        if col in frame.columns:
            future[col] = np.nan
    combined = pd.concat([frame, future], ignore_index=True)
    x = build_features(combined).tail(request.horizon_hours).reindex(columns=bundle["feature_columns"])
    forecast_rows = []
    for i, timestamp in enumerate(future_times):
        row = x.iloc[[i]]
        forecast_rows.append({
            "timestamp": timestamp.isoformat(),
            "q10_eur_mwh": float(bundle["models"]["0.1"].predict(row)[0]),
            "q50_eur_mwh": float(bundle["models"]["0.5"].predict(row)[0]),
            "q90_eur_mwh": float(bundle["models"]["0.9"].predict(row)[0]),
        })
    return {"market": "DE-LU", "training_end": bundle["training_end"], "forecast": forecast_rows}
