from power_price.config import ForecastConfig
from power_price.data import generate_benchmark_data
from power_price.features import build_features
from power_price.modeling import fit_quantile_models

def test_quantile_models_fit():
    data = generate_benchmark_data(1400, 12)
    x = build_features(data)
    valid = x.notna().all(axis=1)
    models = fit_quantile_models(x.loc[valid], data.loc[valid, "price_eur_mwh"], ForecastConfig())
    assert set(models) == {"0.1", "0.5", "0.9"}
    assert all(len(models[q].predict(x.loc[valid].tail(5))) == 5 for q in models)
