# European Electricity Price Forecasting Platform

A production-oriented forecasting service for hourly day-ahead wholesale electricity prices. The platform combines leakage-aware feature engineering, LightGBM quantile regression, chronological walk-forward evaluation, benchmark models, forecast artifact versioning, and a typed inference API.

## Forecast contract
- Market: German-Luxembourg bidding zone (DE-LU), configurable through the data contract.
- Target: hourly day-ahead wholesale price in EUR/MWh.
- Horizon: next 24 hourly delivery periods.
- Point forecast: median (q50); uncertainty: q10/q90 quantile forecasts.
- Primary estimator: LightGBM quantile regression. Linear and persistence baselines remain in evaluation; model complexity must earn its place through backtesting.

## Architecture
```text
Market prices (CSV / Energy-Charts adapter)
 -> schema validation and timestamp alignment
 -> calendar + lagged price + optional known-at-issue covariates
 -> daily walk-forward evaluation: persistence / Ridge / LightGBM quantiles
 -> model bundle + metrics + 24-hour forecast API
```

## Quick start
Python 3.10+:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
power-price benchmark --rows 26280 --output artifacts
```
The benchmark generator is synthetic. Its scores validate engineering behavior, not market performance.

## Train against a normalized market CSV
Required columns: `timestamp`, `price_eur_mwh`. Optional covariates: `load_forecast_mw`, `wind_forecast_mw`, `solar_forecast_mw`, `net_import_forecast_mw`.
```bash
power-price train --input data/market_prices.csv --output artifacts
```
Optional covariates must be forecasts or values available at the forecast issue time. Realized future load/generation must never be joined as if it were known on day D-1.

## Retrieve public historical prices
```bash
power-price fetch-energy-charts --start 2023-01-01 --end 2025-12-31 --zone DE-LU --output data/de_lu_prices.csv
power-price train --input data/de_lu_prices.csv --output artifacts
```
The Energy-Charts adapter is a public-source integration, not a guaranteed production SLA. Check current API terms, resolution and completeness; commercial operation may require licensed data and source reconciliation.

## Run the API
```bash
export POWER_PRICE_MODEL_PATH=artifacts/model_bundle.joblib
uvicorn power_price.api:app --host 0.0.0.0 --port 8000
```
Interactive contract: `http://127.0.0.1:8000/docs`. POST `/forecast` accepts at least 504 hourly history records and returns q10/q50/q90 for up to the next 24 hours.

## Evaluation
Daily rolling origins; model fit uses only data before each origin. Metrics include MAE, RMSE, sMAPE, quantile pinball loss and q10–q90 empirical coverage. Baselines are previous-day same-hour persistence and regularized linear regression.

## Operational limitations
Electricity prices have spikes, negative prices, structural breaks and regime changes. Quantile forecasts do not guarantee calibrated tails during unprecedented regimes. Production use requires feature-freshness checks, spike/regime monitoring, retraining controls, data snapshots, versioned models and a fallback. This repository does not claim trading profitability; decision value depends on market rules, transaction costs, imbalance exposure and risk constraints.

## Container
```bash
docker build -t electricity-price-forecasting .
docker run --rm -p 8000:8000 -e POWER_PRICE_MODEL_PATH=/app/artifacts/model_bundle.joblib -v "$PWD/artifacts:/app/artifacts:ro" electricity-price-forecasting
```
