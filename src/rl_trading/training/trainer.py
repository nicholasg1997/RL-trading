from ..configs.data_classes import OffPolicyConfig
from ..evaluation.metrics import compute_portfolio_metrics
from ..agents.sac import SACAgent
from .buffer import Buffer
from pathlib import Path
import pandas as pd
import numpy as np
from ..configs.stock_universe import ASSET_TICKERS


class OffPolicyTrainer:
	def __init__(self, env, eval_env, agent: SACAgent, buffer: Buffer, config: OffPolicyConfig, logger):
		self.env = env
		self.eval_env = eval_env
		self.agent = agent
		self.buffer = buffer
		self.config = config
		self.logger = logger

		self.warmup_steps = config.warmup_steps
		self.update_interval = config.update_interval
		self.eval_interval = config.eval_interval

	def train(self, total_steps: int):
		episode_reward = 0.0
		episode_length = 0
		episode_stats = self._reset_episode_stats()

		env = self.env

		obs, _ = env.reset()

		for step in range(1, total_steps+1):
			if step < self.warmup_steps:
				action = env.action_space.sample()
			else:
				action = self.agent.select_action(obs)

			next_obs, rew, terminated, truncated, info = env.step(action)
			self.buffer.add(obs, action, rew, next_obs, terminated, truncated)

			episode_reward += rew
			episode_length += 1
			self._record_train_step(episode_stats, info)

			if terminated or truncated:
				self.logger.log_scalar("train/episode_reward", episode_reward, step)
				self.logger.log_scalar("train/episode_length", episode_length, step)
				self._log_train_episode(episode_stats, step)
				episode_reward = 0.0
				episode_length = 0
				episode_stats = self._reset_episode_stats()
				obs, _ = env.reset()
			else:
				obs = next_obs

			if (step >= self.config.update_after and
				step % self.update_interval == 0 and
				len(self.buffer) >= self.config.batch_size):
				for _ in range(self.config.update_every):
					self._update(step)

			if self.eval_interval and step % self.eval_interval == 0:
				eval_metrics = self.evaluate()
				self._log_eval_metrics(eval_metrics, step)
				self._save_best_if_needed(eval_metrics, step)

			if self.config.save_interval and step % self.config.save_interval == 0:
				self.agent.save(Path(self.config.checkpoint_dir) / f"step_{step}", step=step)

			if step % 10_000 == 0:
				alpha_val = self.agent.alpha.item()
				print(f"Step {step}/{total_steps} | Alpha: {alpha_val:.4f}", flush=True)

	def _reset_episode_stats(self) -> dict[str, list[float]]:
		return {
			"raw_reward": [],
			"net_return": [],
			"portfolio_value": [],
			"drawdown": [],
			"turnover": [],
			"transaction_cost": [],
			"gross_exposure": [],
			"reward_log_return": [],
			"reward_downside_penalty": [],
			"reward_drawdown_penalty": [],
			"reward_turnover_penalty": [],
		}

	def _record_train_step(self, episode_stats: dict[str, list[float]], info: dict) -> None:
		for key in episode_stats:
			if key in info:
				episode_stats[key].append(float(info[key]))

	def _log_train_episode(self, episode_stats: dict[str, list[float]], step: int) -> None:
		if not episode_stats["portfolio_value"]:
			return

		portfolio_values = np.asarray(episode_stats["portfolio_value"], dtype=np.float64)
		net_returns = np.asarray(episode_stats["net_return"], dtype=np.float64)
		turnover = np.asarray(episode_stats["turnover"], dtype=np.float64)
		transaction_cost = np.asarray(episode_stats["transaction_cost"], dtype=np.float64)
		gross_exposure = np.asarray(episode_stats["gross_exposure"], dtype=np.float64)

		self.logger.log_scalar("train/episode_final_value", float(portfolio_values[-1]), step)
		self.logger.log_scalar("train/episode_portfolio_return", float(portfolio_values[-1] - 1.0), step)
		self.logger.log_scalar("train/episode_mean_net_return", float(net_returns.mean()), step)
		self.logger.log_scalar("train/episode_net_return_std", float(net_returns.std(ddof=1)) if len(net_returns) > 1 else 0.0, step)
		self.logger.log_scalar("train/episode_max_drawdown", float(max(episode_stats["drawdown"])), step)
		self.logger.log_scalar("train/episode_avg_turnover", float(turnover.mean()), step)
		self.logger.log_scalar("train/episode_total_transaction_cost", float(transaction_cost.sum()), step)
		self.logger.log_scalar("train/episode_avg_gross_exposure", float(gross_exposure.mean()), step)

		for key in (
			"raw_reward",
			"reward_log_return",
			"reward_downside_penalty",
			"reward_drawdown_penalty",
			"reward_turnover_penalty",
		):
			values = episode_stats[key]
			if values:
				self.logger.log_scalar(f"train/episode_mean_{key}", float(np.mean(values)), step)


	def _update(self, step: int):
		batch = self.buffer.sample(self.config.batch_size)
		metrics = self.agent.update(batch)
		if step % 1000 == 0:
			self.logger.log_dict(metrics, step)

	def evaluate(self) -> dict[str, float]:
		env = self.eval_env
		agent = self.agent

		obs, info = env.reset()

		# We start tracking history, beginning with the starting state (t0)
		# to ensure initial portfolio value and date are aligned.
		records = [{
			"date": env.dates[env._t0],
			"gross_return": 0.0,
			"net_return": 0.0,
			"portfolio_value": 1.0,
			"turnover": 0.0,
			"transaction_cost": 0.0,
			"weights": np.zeros(env.n_assets),
			"benchmark_return": 0.0,
			"gross_exposure": 0.0,
		}]

		truncated = False
		terminated = False

		while not (terminated or truncated):
			# Select deterministic actions during evaluation (no exploration noise)
			action = agent.select_action(obs, deterministic=True)

			obs, reward, terminated, truncated, info = env.step(action)

			# Map environment keys to the baseline / metrics schema
			records.append({
				"date": info["date"],
				"gross_return": info["portfolio_return"],  # environment portfolio_return is gross return
				"net_return": info["net_return"],
				"portfolio_value": info["portfolio_value"],
				"turnover": info["turnover"],
				"transaction_cost": info["transaction_cost"],
				"weights": info["weights"],
				"benchmark_return": info["benchmark_return"],
				"gross_exposure": info["gross_exposure"],  # tracked for logger
			})

		# Convert steps to DataFrame
		df = pd.DataFrame(records)

		# Calculate standard metrics using metrics.py
		metrics = compute_portfolio_metrics(df)

		# Map raw metrics to the exact keys expected by logger & config.best_metric
		eval_metrics = {
			"validation/sharpe": metrics["sharpe_ratio"],
			"validation/sortino": metrics["sortino_ratio"],
			"validation/calmar": metrics["calmar_ratio"],
			"validation/max_drawdown": metrics["max_drawdown"],
			"validation/portfolio_return": metrics["annualized_return"],
			"validation/final_value": metrics["final_value"],
			"validation/annualized_volatility": metrics["annualized_volatility"],
			"validation/win_rate": metrics["win_rate"],
			"validation/average_turnover": metrics["avg_daily_turnover"],
			"validation/average_gross_exposure": float(df["gross_exposure"].mean()),
			"validation/transaction_costs": metrics["total_transaction_costs"],
		}

		benchmark_returns = pd.Series(df["benchmark_return"].to_numpy(dtype=np.float64))
		benchmark_value = float((1.0 + benchmark_returns).prod())
		n_days = len(df)
		benchmark_annualized_return = float(benchmark_value ** (252 / n_days) - 1.0)
		eval_metrics["validation/benchmark_return"] = benchmark_annualized_return
		eval_metrics["validation/benchmark_final_value"] = benchmark_value
		eval_metrics["validation/excess_annualized_return"] = (
			eval_metrics["validation/portfolio_return"] - benchmark_annualized_return
		)

		weight_matrix = np.stack(df["weights"].values)  # shape (T, n_assets)
		mean_weights = weight_matrix.mean(axis=0)  # average allocation per asset

		for i, ticker in enumerate(ASSET_TICKERS):
			eval_metrics[f"validation/weight_{ticker}"] = float(mean_weights[i])

		# Also log weight concentration (Herfindahl index) — useful for detecting over-concentration
		herfindahl = float(np.sum(mean_weights ** 2))
		eval_metrics["validation/weight_concentration"] = herfindahl

		return eval_metrics
	def _log_eval_metrics(self, metrics: dict[str, float], step: int):
		if not hasattr(self.logger, "log_scalar"):
			return
		for name, value in metrics.items():
			self.logger.log_scalar(name, value, step)

	def _save_best_if_needed(self, metrics: dict[str, float], step: int):
		score = metrics.get(self.config.best_metric)
		if score is None:
			return
		if not hasattr(self, "best_eval_score") or score > self.best_eval_score:
			self.best_eval_score = score
			self.agent.save(Path(self.config.checkpoint_dir) / "best", step=step, metrics=metrics)

	def save(self, path):
		self.agent.save(path)

	def load(self, path):
		self.agent.load(path)
