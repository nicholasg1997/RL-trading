from ..configs.data_classes import OffPolicyConfig
from ..evaluation.metrics import compute_portfolio_metrics
from ..agents.sac import SACAgent
from .buffer import Buffer
from pathlib import Path
import pandas as pd
import numpy as np


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

			if terminated or truncated:
				self.logger.log_scalar("train/episode_reward", episode_reward, step)
				self.logger.log_scalar("train/episode_length", episode_length, step)
				episode_reward = 0.0
				episode_length = 0
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
			"validation/max_drawdown": metrics["max_drawdown"],
			"validation/portfolio_return": metrics["annualized_return"],
			"validation/average_turnover": metrics["avg_daily_turnover"],
			"validation/average_gross_exposure": float(df["gross_exposure"].mean()),
			"validation/transaction_costs": metrics["total_transaction_costs"],
		}

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
