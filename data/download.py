import os
from pathlib import Path
import yfinance as yf
import pandas as pd
from yfinance import tickers

from src.rl_trading.configs.stock_universe import ASSET_TICKERS, BENCHMARK_TICKER
from src.rl_trading.data.features import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2000-01-01"
END_DATE = "2025-12-31"

def download_raw_data():
	tickers = ASSET_TICKERS + [BENCHMARK_TICKER]
	for ticker in tickers:
		print(f"Downloading {ticker} data...")
		t = yf.Ticker(ticker)
		df = t.history(start=START_DATE, end=END_DATE, interval="1d")[['Open','High','Low','Close','Volume']]

		if df.empty:
			print(f"WARNING: No data found for {ticker}")
			continue

		csv_path = RAW_DIR / f"{ticker}.csv"
		df.to_csv(csv_path, header=True)
		print(f"Saved {ticker} data to {csv_path}")

def process_raw_data():
	processed = {}
	for ticker in ASSET_TICKERS:
		processed[ticker] = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col="Date", parse_dates=True)

	processed[BENCHMARK_TICKER] = pd.read_csv(RAW_DIR / f"{BENCHMARK_TICKER}.csv", index_col="Date", parse_dates=True)
	engineered = engineer_features(processed)
	aligned_dates = engineered.index

	asset_returns = pd.DataFrame(index=aligned_dates)
	for ticker in ASSET_TICKERS:
		asset_returns[ticker] = processed[ticker]['Close'].pct_change(1)

	benchmark_returns = processed[BENCHMARK_TICKER]['Close'].pct_change(1).reindex(aligned_dates)
	asset_returns = asset_returns.reindex(aligned_dates)

	engineered.to_csv(PROCESSED_DIR / "features.csv")
	asset_returns.to_csv(PROCESSED_DIR / "asset_returns.csv")
	benchmark_returns.to_csv(PROCESSED_DIR / "benchmark_returns.csv")

	print("Successfully processed and saved features and returns!")


if __name__ == "__main__":
	download_raw_data()
	process_raw_data()





