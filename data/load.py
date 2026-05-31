import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_data():
	features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col="Date", parse_dates=True)
	asset_returns = pd.read_csv(PROCESSED_DIR / "asset_returns.csv", index_col="Date", parse_dates=True)
	benchmark_returns = pd.read_csv(PROCESSED_DIR / "benchmark_returns.csv", index_col="Date", parse_dates=True)

	return features, asset_returns, benchmark_returns

if __name__ == "__main__":
	features, asset_returns, benchmark_returns = load_data()
	print(len(features), len(asset_returns), len(benchmark_returns))