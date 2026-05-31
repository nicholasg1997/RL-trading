import numpy as np
import pandas as pd


def sharpe_ratio(net_returns: pd.Series, risk_free_rate_daily: float = 0.0, periods_per_year: int = 252) -> float:
	"""
	Computes the annualized Sharpe Ratio.
	"""
	excess_returns = net_returns - risk_free_rate_daily
	mean_excess = excess_returns.mean()
	std_excess = excess_returns.std(ddof=1)

	if std_excess == 0 or np.isnan(std_excess):
		return 0.0

	return float((mean_excess / std_excess) * np.sqrt(periods_per_year))


def sortino_ratio(net_returns: pd.Series, risk_free_rate_daily: float = 0.0, periods_per_year: int = 252) -> float:
	"""
	Computes the annualized Sortino Ratio (penalizes only downside deviation).
	"""
	excess_returns = net_returns - risk_free_rate_daily
	mean_excess = excess_returns.mean()

	downside_returns = np.minimum(excess_returns, 0.0)
	downside_std = np.sqrt(np.mean(downside_returns ** 2))

	if downside_std == 0 or np.isnan(downside_std):
		return 0.0

	return float((mean_excess / downside_std) * np.sqrt(periods_per_year))


def max_drawdown(portfolio_values: pd.Series) -> float:
	"""
	Computes the maximum drawdown from a series of portfolio values.
	"""
	if len(portfolio_values) == 0:
		return 0.0

	rolling_max = portfolio_values.cummax()
	drawdowns = (rolling_max - portfolio_values) / rolling_max
	return float(drawdowns.max())


def compute_portfolio_metrics(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
	"""
	Takes the output DataFrame from a baseline run or agent step collection,
	and returns a summary dictionary of all core performance metrics.
	"""
	net_ret = df["net_return"]
	p_val = df["portfolio_value"]

	# Calculate Annualized Return (Compound Annual Growth Rate)
	n_days = len(df)
	total_return = p_val.iloc[-1] / p_val.iloc[0] if p_val.iloc[0] > 0 else p_val.iloc[-1]
	ann_return = float(total_return ** (periods_per_year / n_days) - 1.0)

	ann_vol = float(net_ret.std(ddof=1) * np.sqrt(periods_per_year))

	mdd = max_drawdown(p_val)
	calmar = ann_return / mdd if mdd > 0 else 0.0

	sharpe = sharpe_ratio(net_ret, periods_per_year=periods_per_year)
	sortino = sortino_ratio(net_ret, periods_per_year=periods_per_year)

	win_rt = float((net_ret > 0).mean())

	total_turnover = float(df["turnover"].sum())
	avg_turnover = float(df["turnover"].mean())
	total_costs = float(df["transaction_cost"].sum())

	return {
		"annualized_return": ann_return,
		"annualized_volatility": ann_vol,
		"sharpe_ratio": sharpe,
		"sortino_ratio": sortino,
		"max_drawdown": mdd,
		"calmar_ratio": calmar,
		"win_rate": win_rt,
		"total_turnover": total_turnover,
		"avg_daily_turnover": avg_turnover,
		"total_transaction_costs": total_costs,
		"final_value": float(p_val.iloc[-1])
	}


def run_agent_evaluation(env, agent) -> pd.DataFrame:
	"""
	Runs a single deterministic evaluation episode with the RL agent
	and returns a DataFrame formatted identically to the baseline outputs.
	"""
	obs, info = env.reset()
	records = []

	# Track the initial state (before trades occur on t=0)
	# This aligns the dates and sets up the starting portfolio value of 1.0
	records.append({
		"date": env.dates[env._t0],
		"gross_return": 0.0,
		"net_return": 0.0,
		"portfolio_value": 1.0,
		"turnover": 0.0,
		"transaction_cost": 0.0,
		"weights": np.zeros(env.n_assets),
		"benchmark_return": 0.0,
	})

	truncated = False
	terminated = False

	while not (terminated or truncated):
		action = agent.get_action(obs, deterministic=True)

		obs, reward, terminated, truncated, info = env.step(action)

		records.append({
			"date": info["date"],
			"gross_return": info["portfolio_return"],
			"net_return": info["net_return"],
			"portfolio_value": info["portfolio_value"],
			"turnover": info["turnover"],
			"transaction_cost": info["transaction_cost"],
			"weights": info["weights"],
			"benchmark_return": info["benchmark_return"],
		})

	return pd.DataFrame(records)