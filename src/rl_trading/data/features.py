import pandas as pd

EPSILON = 1e-8

def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
	delta = close.diff()
	gain = delta.clip(lower=0).rolling(window).mean()
	loss = (-delta.clip(upper=0)).rolling(window).mean()
	rs = gain/(loss + EPSILON)
	rsi = 100 - (100 / (1+rs))
	return (rsi - 50) / 50

def calculate_macd(close: pd.Series, fast_window: int = 12, slow_window: int = 26, signal_period: int = 9, normalize=True) -> tuple[pd.Series, pd.Series]:
	ema_fast = close.ewm(span=fast_window, adjust=False).mean()
	ema_slow = close.ewm(span=slow_window, adjust=False).mean()
	macd_line = (ema_fast - ema_slow)
	if normalize:
		macd_line /= close
	signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
	return macd_line, macd_line - signal_line

def calculate_bb(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
	sma = close.rolling(window).mean()
	std = close.rolling(window).std()
	upper_band = sma + (2 * std)
	lower_band = sma - (2 * std)
	return sma, upper_band, lower_band

def calculate_bb_extras(close, sma, upper_band, lower_band):
	percentage_b = (close-lower_band)/(upper_band-lower_band + EPSILON)
	return percentage_b

def calculate_atr(high, low, close, window: int = 14, normalize=True) -> pd.Series:
	tr = pd.concat([
		high - low,
		(high - close.shift(1)).abs(),
		(low - close.shift(1)).abs()
	], axis=1).max(axis=1)
	atr = tr.rolling(window).mean()
	return atr / close if normalize else atr

def calculate_volume_ratio(volume: pd.Series, window: int = 14) -> pd.Series:
	return (volume / (volume.rolling(window).mean() + EPSILON)).clip(0, 5)

def engineer_features(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
	feature_blocks = []
	for ticker, df in prices.items():
		close = df['Close']
		high = df['High']
		low = df['Low']
		vol = df['Volume']

		sma, upper_band, lower_band = calculate_bb(close, window=20)
		bb_position = calculate_bb_extras(close, sma, upper_band, lower_band)
		macd_line, histogram = calculate_macd(close, normalize=True)
		price_pos = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + EPSILON)

		block = pd.DataFrame({
			f"{ticker}_return_1d": close.pct_change(1),
			f"{ticker}_return_5d": close.pct_change(5),
			f"{ticker}_return_20d": close.pct_change(20),
			f"{ticker}_rsi_14": calculate_rsi(close, 14),
			f"{ticker}_macd_line": macd_line,
			f"{ticker}_macd_histogram": histogram,
			f"{ticker}_bb_position": bb_position,
			f"{ticker}_price_pos_20": price_pos,
			f"{ticker}_atr_normalized": calculate_atr(high, low, close, normalize=True),
			f"{ticker}_volume_ratio": calculate_volume_ratio(vol, window=14),
			
		})

		feature_blocks.append(block)

	features = pd.concat(feature_blocks, axis=1)
	#cross asset correlation
	returns = pd.DataFrame({
		t: df['Close'].pct_change(1)
		for t, df in prices.items()
	})
	tickers = list(prices.keys())

	for i, t1 in enumerate(tickers):
		for t2 in tickers[i+1:]:
			col = f"corr_{t1}_{t2}"
			features[col] = returns[t1].rolling(20).corr(returns[t2])

	features = features.shift(1)
	features = features.dropna()
	return features
