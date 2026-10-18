import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def point_metrics(actual, predicted):
    """Return point-forecast metrics robust to zero and negative electricity prices."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return {
        "mae_eur_mwh": float(mean_absolute_error(actual, predicted)),
        "rmse_eur_mwh": float(np.sqrt(mean_squared_error(actual, predicted))),
        "smape_pct": float(np.mean(np.divide(2*np.abs(actual-predicted), np.abs(actual)+np.abs(predicted),
            out=np.zeros_like(actual), where=(np.abs(actual)+np.abs(predicted))>1e-9))*100),
    }

def pinball_loss(actual, predicted, quantile):
    """Score probabilistic forecasts using asymmetric quantile loss."""
    error = np.asarray(actual, float)-np.asarray(predicted, float)
    return float(np.mean(np.maximum(quantile*error, (quantile-1)*error)))

def interval_metrics(actual, lower, upper):
    """Measure empirical interval coverage and average width."""
    actual, lower, upper = np.asarray(actual), np.asarray(lower), np.asarray(upper)
    return {"coverage_pct": float(np.mean((actual>=lower)&(actual<=upper))*100),
            "mean_width_eur_mwh": float(np.mean(upper-lower))}
