import argparse
import json
from pathlib import Path
from power_price.backtest import walk_forward_backtest
from power_price.config import ForecastConfig
from power_price.data import fetch_energy_charts_prices, generate_benchmark_data, load_market_csv
from power_price.modeling import train_bundle

def main():
    parser = argparse.ArgumentParser(description="European electricity price forecasting platform")
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--rows", type=int, default=26280)
    bench.add_argument("--output", default="artifacts")
    bench.add_argument("--test-days", type=int, default=60)
    train = commands.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", default="artifacts")
    train.add_argument("--test-days", type=int, default=60)
    fetch = commands.add_parser("fetch-energy-charts")
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--zone", default="DE-LU")
    fetch.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "fetch-energy-charts":
        data = fetch_energy_charts_prices(args.start, args.end, args.zone)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"Saved {len(data)} price records to {path}")
        return

    config = ForecastConfig(test_days=args.test_days, output_dir=Path(args.output))
    frame = generate_benchmark_data(args.rows) if args.command == "benchmark" else load_market_csv(args.input)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    report = walk_forward_backtest(frame, output, config)
    bundle = train_bundle(frame, output/"model_bundle.joblib", config)
    (output/"run_metadata.json").write_text(json.dumps({
        "source_rows": len(frame), "training_end": bundle["training_end"],
        "training_rows": bundle["training_rows"], "config": config.to_dict(),
        "evaluation_protocol": report["protocol"],
    }, indent=2))
    print(json.dumps(report["metrics"], indent=2))
    print(f"Artifacts written to {output.resolve()}")

if __name__ == "__main__":
    main()
