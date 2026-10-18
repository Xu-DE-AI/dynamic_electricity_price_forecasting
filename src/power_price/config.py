from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class ForecastConfig:
    """Versioned defaults for a 24-hour day-ahead forecast."""
    horizon_hours: int = 24
    lookback_min_rows: int = 24 * 14
    test_days: int = 60
    random_state: int = 41
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    output_dir: Path = Path("artifacts")

    def to_dict(self):
        value = asdict(self)
        value["output_dir"] = str(value["output_dir"])
        value["quantiles"] = list(value["quantiles"])
        return value

OPTIONAL_COVARIATES = ["load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw", "net_import_forecast_mw"]
