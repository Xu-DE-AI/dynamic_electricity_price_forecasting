from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from power_price.config import ForecastConfig
from power_price.features import build_features
from power_price.metrics import interval_metrics, pinball_loss, point_metrics

def fit_quantile_models(features: pd.DataFrame, target: pd.Series, config: ForecastConfig):
    """Fit lower, median and upper LightGBM quantile estimators."""
    models = {}
    for quantile in config.quantiles:
        models[str(quantile)] = LGBMRegressor(
            objective="quantile", alpha=quantile, n_estimators=500, learning_rate=.035,
            num_leaves=31, min_child_samples=35, subsample=.85, subsample_freq=1,
            colsample_bytree=.85, reg_lambda=2.0, random_state=config.random_state,
            verbosity=-1, n_jobs=-1,
        ).fit(features, target)
    return models

def fit_linear_baseline(features, target):
    """Train a regularized linear baseline with median imputation and scaling."""
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    return model.fit(features, target)

def train_bundle(frame: pd.DataFrame, output_path, config: ForecastConfig | None = None):
    """Fit final deployable models and persist the feature contract and training horizon."""
    config = config or ForecastConfig()
    data = frame.sort_values("timestamp").reset_index(drop=True)
    x, y = build_features(data), data["price_eur_mwh"].astype(float)
    valid = x.notna().all(axis=1)
    x, y = x.loc[valid], y.loc[valid]
    if len(x) < config.lookback_min_rows:
        raise ValueError(f"Insufficient feature-complete rows: {len(x)}")
    models = fit_quantile_models(x, y, config)
    bundle = {"models": models, "feature_columns": list(x.columns),
              "training_end": str(data["timestamp"].max()), "training_rows": int(len(x)),
              "quantiles": list(config.quantiles), "config": config.to_dict()}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle

def evaluate_predictions(actual, predictions):
    """Build a consistent point and probabilistic scorecard."""
    actual = np.asarray(actual, float)
    q10, q50, q90 = (np.asarray(predictions[k], float) for k in ("0.1", "0.5", "0.9"))
    return {"median": point_metrics(actual, q50),
            "pinball_q10": pinball_loss(actual, q10, .1),
            "pinball_q50": pinball_loss(actual, q50, .5),
            "pinball_q90": pinball_loss(actual, q90, .9),
            "interval_q10_q90": interval_metrics(actual, q10, q90)}
