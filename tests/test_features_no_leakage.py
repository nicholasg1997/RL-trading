import pandas as pd

def test_no_leakage(synthetic_prices: dict[str, pd.DataFrame]):
	from rl_trading.data.features import engineer_features
	import copy

	features_original = engineer_features(copy.deepcopy(synthetic_prices))

	corrupted = copy.deepcopy(synthetic_prices)
	for df in corrupted.values():
		df.iloc[-10:] *= 999.0

	featured_corrupted = engineer_features(corrupted)

	pd.testing.assert_frame_equal(features_original[:-10],
	                              featured_corrupted[:-10])

def test_no_nans_after_warmup(synthetic_features: pd.DataFrame):
	assert synthetic_features.isna().sum().sum() == 0

def test_feature_dimensions(synthetic_features: pd.DataFrame):
	assert synthetic_features.shape[1] == 33
