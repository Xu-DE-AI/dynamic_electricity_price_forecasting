import json
from pathlib import Path
import numpy as np
import pandas as pd

from power_price.config import ForecastConfig
from power_price.features import build_features
from power_price.metrics import interval_metrics, pinball_loss, point_metrics
from power_price.modeling import fit_linear_baseline, fit_quantile_models

def walk_forward_backtest(frame: pd.DataFrame, output_dir, config: ForecastConfig | None = None):
    """Score daily 24-hour forecasts, fitting each model only on pre-origin observations."""
    config = config or ForecastConfig()
    data = frame.sort_values("timestamp").reset_index(drop=True)
    x, y = build_features(data), frame.sort_values("timestamp").reset_index(drop=True)["price_eur_mwh"].astype(float)
    first_test = max(config.lookback_min_rows, len(data)-config.test_days*config.horizon_hours)
    origins = list(range(first_test, len(data), config.horizon_hours))
    rows, origin_records = [], []
    for origin in origins:
        end = min(origin+config.horizon_hours, len(data))
        train_mask = (np.arange(len(data)) < origin) & x.notna().all(axis=1).to_numpy()
        if train_mask.sum() < config.lookback_min_rows:
            continue
        train_x, train_y = x.loc[train_mask], y.loc[train_mask]
        test_x, test_y = x.iloc[origin:end], y.iloc[origin:end]
        # A fresh fit at each origin mimics a daily retraining/evaluation cadence.
        models = fit_quantile_models(train_x, train_y, config)
        linear = fit_linear_baseline(train_x, train_y)
        qpred = {key: model.predict(test_x) for key, model in models.items()}
        linear_pred = linear.predict(test_x)
        # Previous-day same-hour price is available at the day-ahead issue time.
        persistence = y.iloc[origin-24:origin-24+len(test_x)].to_numpy(float)
        for j, idx in enumerate(range(origin, end)):
            rows.append({"timestamp": data.timestamp.iloc[idx], "actual_price_eur_mwh": float(y.iloc[idx]),
                "q10_eur_mwh": float(qpred["0.1"][j]), "q50_eur_mwh": float(qpred["0.5"][j]),
                "q90_eur_mwh": float(qpred["0.9"][j]), "linear_eur_mwh": float(linear_pred[j]),
                "persistence_24h_eur_mwh": float(persistence[j]), "forecast_origin": str(data.timestamp.iloc[origin])})
        origin_records.append({"origin": str(data.timestamp.iloc[origin]), "hours": len(test_x)})
    if not rows:
        raise ValueError("No valid backtest folds; increase history or reduce test_days")
    scored = pd.DataFrame(rows)
    actual = scored.actual_price_eur_mwh.to_numpy(float)
    metrics = {
        "lightgbm_q50": point_metrics(actual, scored.q50_eur_mwh),
        "linear_ridge": point_metrics(actual, scored.linear_eur_mwh),
        "persistence_24h": point_metrics(actual, scored.persistence_24h_eur_mwh),
        "probabilistic": {
            "pinball_q10": pinball_loss(actual, scored.q10_eur_mwh, .1),
            "pinball_q50": pinball_loss(actual, scored.q50_eur_mwh, .5),
            "pinball_q90": pinball_loss(actual, scored.q90_eur_mwh, .9),
            "q10_q90_interval": interval_metrics(actual, scored.q10_eur_mwh, scored.q90_eur_mwh),
        },
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output/"walk_forward_predictions.csv", index=False)
    report = {"protocol": "daily rolling-origin; 24-hour delivery horizon", "forecast_origins": len(origin_records),
              "scored_hours": len(scored), "test_start": str(scored.timestamp.min()),
              "test_end": str(scored.timestamp.max()), "metrics": metrics, "origins": origin_records}
    (output/"backtest_report.json").write_text(json.dumps(report, indent=2))
    return report
