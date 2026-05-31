from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseBaseline(ABC):

	def __init__(self,
	             asset_returns: np.ndarray,
	             benchmark_returns: np.ndarray,
	             dates: pd.DatetimeIndex,
	             transaction_cost_rate: float = 0.001):
		self.asset_returns = asset_returns
		self.benchmark_returns = benchmark_returns
		self.dates = dates
		self.transaction_cost_rate = transaction_cost_rate
		self.n_assets = asset_returns.shape[1]

	@abstractmethod
	def get_weights(self, t: int, drifted_weights:np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def run(self) -> pd.DataFrame:
		T = len(self.asset_returns)
		active_weights = np.zeros(self.n_assets)
		portfolio_value = 1.0
		records = []

		for t in range(T):
			if t == 0:
				drifted_weights = np.zeros(self.n_assets)
			else:
				prev_returns = self.asset_returns[t-1]
				denom = 1.0 + np.dot(active_weights, prev_returns)
				if denom > 0:
					drifted_weights = active_weights * (1.0 + prev_returns) / denom
				else:
					drifted_weights = np.zeros(self.n_assets)

			w = self.get_weights(t, drifted_weights)

			if t == 0:
				turnover = float(np.abs(w).sum()) # initial funding cost
			else:
				turnover = float(np.abs(w - drifted_weights).sum())

			cost = turnover * self.transaction_cost_rate
			gross_return = float(np.dot(w, self.benchmark_returns[t]))
			net_return = gross_return - cost
			portfolio_value *= (1.0 + net_return)

			records.append({
				"date": self.dates[t],
				"gross_return": gross_return,
				"net_return": net_return,
				"portfolio_value": portfolio_value,
				"turnover": turnover,
				"transaction_cost": cost,
				"weights": w.copy(),
				"benchmark_return": float(self.benchmark_returns[t]),
			})

			active_weights = w.copy()

		return pd.DataFrame(records)


class SPYBuyAndHold(BaseBaseline):

	def get_weights(self, t: int) -> np.ndarray:
		return np.zeros(self.n_assets)

	def run(self) -> pd.DataFrame:
		T = len(self.asset_returns)
		portfolio_value = 1.0
		records = []

		for t in range(T):
			ret = float(self.benchmark_returns[t])
			portfolio_value *= (1.0 + ret)
			records.append({
				"date": self.dates[t],
				"gross_return": ret,
				"net_return": ret,  # no transaction costs for buy-and-hold
				"portfolio_value": portfolio_value,
				"turnover": 0.0,
				"transaction_cost": 0.0,
				"weights": np.zeros(self.n_assets),
				"benchmark_return": ret,
			})

		return pd.DataFrame(records)


class EqualWeightBuyAndHold(BaseBaseline):
	def get_weights(self, t: int, drifted_weights: np.ndarray) -> np.ndarray:
		if t == 0:
			return np.full(self.n_assets, 1.0 / self.n_assets)
		return drifted_weights


class CashBaseline(BaseBaseline):
	def get_weights(self, t: int, drifted_weights: np.ndarray) -> np.ndarray:
		return np.zeros(self.n_assets)

class MonthlyRebalanceEqualWeight(BaseBaseline):
	def get_weights(self, t: int, drifted_weights: np.ndarray) -> np.ndarray:
		if t == 0:
			return np.full(self.n_assets, 1.0 / self.n_assets)

		if self.dates[t].month != self.dates[t-1].month:
			return np.full(self.n_assets, 1.0 / self.n_assets)

		return drifted_weights

class MonthlyMomentumRebalance(BaseBaseline):

	def __init__(self, *args, lookback_period: int = 21, **kwargs):
		super().__init__(*args, **kwargs)
		self.lookback_period = lookback_period

	def get_weights(self, t: int, drifted_weights: np.ndarray) -> np.ndarray:
		if t == 0:
			return np.full(self.n_assets, 1.0 / self.n_assets)

		if self.dates[t].month != self.dates[t-1].month:
			start_idx = max(0, t - self.lookback_period)
			if start_idx == t:
				return np.full(self.n_assets, 1.0 / self.n_assets)

			returns_window = self.asset_returns[start_idx:t]
			cum_returns = np.prod(1.0 + returns_window, axis=0) - 1.0

			best_asset_idx = np.argmax(cum_returns)
			new_weights = np.zeros(self.n_assets)
			new_weights[best_asset_idx] = 1.0
			return new_weights

		return drifted_weights
