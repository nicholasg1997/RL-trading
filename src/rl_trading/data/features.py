import pandas as pd

EPSILON = 1e-8

def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
	"""
	Calculate RSI and scale it from the standard 0-100 range to [-1, 1].

	Args:
		close: Closing prices ordered oldest to newest.
		window: Rolling lookback period for average gains and losses.

	Returns:
		pd.Series: Normalized RSI aligned to the input index.
	"""
	delta = close.diff()
	gain = delta.clip(lower=0).rolling(window).mean()
	loss = (-delta.clip(upper=0)).rolling(window).mean()
	rs = gain/(loss + EPSILON)
	rsi = 100 - (100 / (1+rs))
	return (rsi - 50) / 50

def calculate_macd(close: pd.Series,
	                   fast_window: int = 12,
	                   slow_window: int = 26,
	                   signal_period: int = 9,
	                   normalize: bool = True) -> tuple[pd.Series, pd.Series]:
	"""
	Calculate MACD line and histogram from closing prices.

	Args:
		close: Closing prices ordered oldest to newest.
		fast_window: EMA span for the fast moving average.
		slow_window: EMA span for the slow moving average.
		signal_period: EMA span for the MACD signal line.
		normalize: If True, divide the MACD line by close prices.

	Returns:
		tuple[pd.Series, pd.Series]: MACD line and MACD histogram.
	"""
	ema_fast = close.ewm(span=fast_window, adjust=False).mean()
	ema_slow = close.ewm(span=slow_window, adjust=False).mean()
	macd_line = (ema_fast - ema_slow)
	if normalize:
		macd_line /= close
	signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
	return macd_line, macd_line - signal_line

def calculate_bb(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
	"""
	Calculate Bollinger Band moving average, upper band, and lower band.

	Args:
		close: Closing prices ordered oldest to newest.
		window: Rolling lookback period for the mean and standard deviation.

	Returns:
		tuple[pd.Series, pd.Series, pd.Series]: SMA, upper band, and lower band.
	"""
	sma = close.rolling(window).mean()
	std = close.rolling(window).std()
	upper_band = sma + (2 * std)
	lower_band = sma - (2 * std)
	return sma, upper_band, lower_band

def calculate_bb_extras(close: pd.Series,
                        sma: pd.Series,
                        upper_band: pd.Series,
                        lower_band: pd.Series) -> pd.Series:
	"""
	Calculate Bollinger Band position as percent B.

	Args:
		close: Closing prices ordered oldest to newest.
		sma: Bollinger moving average. Included for call-site symmetry.
		upper_band: Upper Bollinger Band values.
		lower_band: Lower Bollinger Band values.

	Returns:
		pd.Series: Position inside the band, where 0 is lower band and 1 is upper band.
	"""
	percentage_b = (close-lower_band)/(upper_band-lower_band + EPSILON)
	return percentage_b

def calculate_atr(high: pd.Series,
                  low: pd.Series,
                  close: pd.Series,
                  window: int = 14,
                  normalize: bool = True) -> pd.Series:
	"""
	Calculate average true range from high, low, and close prices.

	Args:
		high: Daily high prices ordered oldest to newest.
		low: Daily low prices ordered oldest to newest.
		close: Daily close prices ordered oldest to newest.
		window: Rolling lookback period for average true range.
		normalize: If True, divide ATR by close prices.

	Returns:
		pd.Series: ATR, optionally normalized by close.
	"""
	tr = pd.concat([
		high - low,
		(high - close.shift(1)).abs(),
		(low - close.shift(1)).abs()
	], axis=1).max(axis=1)
	atr = tr.rolling(window).mean()
	return atr / close if normalize else atr

def calculate_volume_ratio(volume: pd.Series, window: int = 14) -> pd.Series:
	"""
	Calculate current volume relative to its rolling average.

	Args:
		volume: Trading volume ordered oldest to newest.
		window: Rolling lookback period for average volume.

	Returns:
		pd.Series: Volume ratio clipped to the range [0, 5].
	"""
	return (volume / (volume.rolling(window).mean() + EPSILON)).clip(0, 5)

def engineer_features(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
	"""
	Build shifted per-asset and cross-asset features for model observations.

	Args:
		prices: Mapping from ticker to OHLCV DataFrame. Each DataFrame must include
			'Close', 'High', 'Low', and 'Volume' columns.

	Returns:
		pd.DataFrame: Feature matrix shifted by one row to avoid lookahead leakage,
		with warmup rows containing missing values removed.
	"""
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
