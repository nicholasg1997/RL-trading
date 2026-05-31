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
	def get_weights(self, t: int) -> np.ndarray:
		raise NotImplementedError

	def run(self) -> pd.DataFrame:
		T = len(self.asset_returns)
		prev_weights = np.zeros(self.n_assets)
		portfolio_value = 1.0
		records = []

		for t in range(T):
			w = self.get_weights(t)
			turnover = float(np.abs(w - prev_weights).sum())
			cost = turnover * self.transaction_cost_rate
			gross_return = float(np.dot(w, self.asset_returns[t]))
			net_return = gross_return - cost
			portfolio_value *= 1.0 + net_return

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

			prev_weights = w.copy()

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
	def get_weights(self, t: int) -> np.ndarray:
		return np.full(self.n_assets, 1.0 / self.n_assets)

class CashBaseline(BaseBaseline):
	def get_weights(self, t: int) -> np.ndarray:
		return np.zeros(self.n_assets)