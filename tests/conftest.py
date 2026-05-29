import pytest
import numpy as np
import pandas as pd

@pytest.fixture(scope="session")
def synthetic_prices():
	n = 300
	dates = pd.date_range("2020-01-01", periods=n, freq="B")
	results = {}
	for ticker, start_price, seed in [
		('ABC', 100, 42),
		('DEF', 10, 43),
		('GHI', 800, 44),
	]:
		rng = np.random.default_rng(seed)
		daily_returns = rng.normal(0.0005, 0.12, size=n)
		close = start_price * np.exp(np.cumsum(daily_returns))
		noise = rng.uniform(0.001, 0.008, size=n)
		high = close + noise
		low = close - noise
		vol = rng.lognormal(14.0, 0.5, size=n)

		results[ticker] = pd.DataFrame({
			"Close": close,
			"High": high,
			"Low": low,
			"Volume": vol,
		}, index=dates)

	return results

@pytest.fixture(scope="session")
def synthetic_features(synthetic_prices: dict[str, pd.DataFrame]):
	from rl_trading.data.features import engineer_features
	return engineer_features(synthetic_prices)


