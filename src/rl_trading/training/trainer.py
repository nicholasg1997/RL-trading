from ..configs.data_classes import OffPolicyConfig
from ..agents.sac import SACAgent
from .buffer import Buffer
from pathlib import Path


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
		raise NotImplementedError(
			"Trading evaluation is not implemented yet. It should run deterministic "
			"validation episodes and return metrics such as validation/sharpe, "
			"validation/max_drawdown, validation/portfolio_return, average_turnover, "
			"average_gross_exposure, and transaction_costs."
		)

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
