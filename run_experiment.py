from __future__ import annotations

import argparse
import yaml

from src.pipelines import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EUR/USD ML experiment pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--fred-key", default="demo", help="FRED API key. For the current configuration it is not required when macro data is disabled.")
    parser.add_argument("--short", action="store_true", help="Use the short 2-year evaluation window from config.yaml.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = run_experiment(cfg, fred_api_key=args.fred_key, use_short_period=args.short)
    print("Experiment finished successfully.")
    print("Artifacts written under results/ and data/.")
    print("Best model:", result["results_table"][0]["model"] if result.get("results_table") else "n/a")
