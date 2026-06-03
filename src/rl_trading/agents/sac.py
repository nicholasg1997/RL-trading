from .gaussian import GaussianActor
from .q_agent import DoubleQCritic
from dataclasses import asdict
import math
import numpy as np
from pathlib import Path
import warnings

import torch

from ..configs.data_classes import ModelParameters, Transition


CHECKPOINT_FILE = "checkpoint.pt"


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
	if isinstance(device, torch.device):
		return device

	requested_device = "auto" if device is None else str(device).lower()
	if requested_device == "auto":
		if torch.cuda.is_available():
			return torch.device("cuda")
		if torch.backends.mps.is_available():
			return torch.device("mps")
		return torch.device("cpu")

	if requested_device == "mps" and not torch.backends.mps.is_available():
		warnings.warn("MPS was requested but is not available. Falling back to CPU.", RuntimeWarning)
		return torch.device("cpu")

	if requested_device.startswith("cuda") and not torch.cuda.is_available():
		warnings.warn("CUDA was requested but is not available. Falling back to CPU.", RuntimeWarning)
		return torch.device("cpu")

	return torch.device(requested_device)


class SACAgent:
	def __init__(self,
	             obs_size: int,
	             action_size: int,
	             config: ModelParameters | None = None,
	             device: str | torch.device | None = "auto"):
		self.obs_size = obs_size
		self.action_size = action_size
		self.config = config or ModelParameters()
		self.device = resolve_device(device)

		self.actor = GaussianActor(
			obs_size,
			action_size,
			hidden_dims=self.config.hidden_dims,
			log_std_min=self.config.log_std_min,
			log_std_max=self.config.log_std_max,
		).to(self.device)
		self.critic = DoubleQCritic(obs_size, action_size, hidden_dims=self.config.hidden_dims).to(self.device)
		self.target_critic = DoubleQCritic(obs_size, action_size, hidden_dims=self.config.hidden_dims).to(self.device)

		self.target_critic.load_state_dict(self.critic.state_dict())

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

		self.log_alpha = torch.nn.Parameter(
			torch.tensor([math.log(self.config.initial_alpha)], dtype=torch.float32, device=self.device)
		)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
		self.target_entropy = self.config.target_entropy
		if self.target_entropy is None:
			self.target_entropy = -float(action_size)

	@property
	def alpha(self) -> torch.Tensor:
		return self.log_alpha.exp()

	def select_action(self, obs, deterministic=False) -> np.ndarray:
		obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
		is_single_obs = obs.ndim == 1
		if is_single_obs:
			obs = obs.unsqueeze(0)

		with torch.no_grad():
			action, _ = self.actor(obs, deterministic)

		action = action.cpu().numpy()
		return action[0] if is_single_obs else action

	def update(self, transitions: Transition):
		rew = transitions.rew.to(self.device)
		next_obs = transitions.next_obs.to(self.device)
		terminated = transitions.terminated.to(self.device)
		truncated = transitions.truncated.to(self.device)
		actions = transitions.act.to(self.device)
		obs = transitions.obs.to(self.device)

		with torch.no_grad():
			next_action, next_log_prob = self.actor(next_obs)
			q1_next, q2_next = self.target_critic(next_obs, next_action)
			q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
			target_q = rew + self._discount_mask(terminated, truncated) * self.config.gamma * q_next

		q1, q2 = self.critic(obs, actions)
		critic_loss = torch.mean((q1 - target_q) ** 2) + torch.mean((q2 - target_q) ** 2)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		new_action, log_prob = self.actor(obs)
		q1_new, q2_new = self.critic(obs, new_action)
		q_new = torch.min(q1_new, q2_new)

		actor_loss = (self.alpha * log_prob - q_new).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self._soft_update()

		return {
			'critic_loss': critic_loss.item(),
			'actor_loss': actor_loss.item(),
			'alpha_loss': alpha_loss.item(),
			'alpha': self.alpha.item()
		}

	def _discount_mask(self, terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
		done = torch.clamp(terminated + truncated, max=1.0)
		return 1.0 - done

	def _soft_update(self):
		for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

	def save(self, path, step: int | None = None, metrics: dict[str, float] | None = None) -> Path:
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)
		checkpoint_path = path / CHECKPOINT_FILE
		torch.save({
			"obs_size": self.obs_size,
			"action_size": self.action_size,
			"config": asdict(self.config),
			"target_entropy": self.target_entropy,
			"step": step,
			"metrics": metrics or {},
			"actor": self.actor.state_dict(),
			"critic": self.critic.state_dict(),
			"target_critic": self.target_critic.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"alpha_optimizer": self.alpha_optimizer.state_dict(),
			"log_alpha": self.log_alpha.detach().cpu(),
		}, checkpoint_path)
		return checkpoint_path

	def load(self, path) -> dict:
		path = Path(path)
		checkpoint_path = path if path.is_file() else path / CHECKPOINT_FILE
		if not checkpoint_path.exists():
			return self._load_legacy_checkpoint(path)

		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		if checkpoint["obs_size"] != self.obs_size or checkpoint["action_size"] != self.action_size:
			raise ValueError(
				"Checkpoint dimensions do not match this agent: "
				f"checkpoint=({checkpoint['obs_size']}, {checkpoint['action_size']}), "
				f"agent=({self.obs_size}, {self.action_size})"
			)

		self.actor.load_state_dict(checkpoint["actor"])
		self.critic.load_state_dict(checkpoint["critic"])
		self.target_critic.load_state_dict(checkpoint["target_critic"])
		self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
		self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
		self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
		self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device).reshape_as(self.log_alpha))
		return {
			"step": checkpoint.get("step"),
			"metrics": checkpoint.get("metrics", {}),
			"config": checkpoint.get("config", {}),
		}

	def load_best(self, checkpoint_dir) -> dict:
		return self.load(Path(checkpoint_dir) / "best")

	def _load_legacy_checkpoint(self, path: Path) -> dict:
		self.actor.load_state_dict(torch.load(path / 'actor.pt', map_location=self.device))
		self.critic.load_state_dict(torch.load(path / 'critic.pt', map_location=self.device))
		target_critic_path = path / 'target_critic.pt'
		if target_critic_path.exists():
			self.target_critic.load_state_dict(torch.load(target_critic_path, map_location=self.device))
		else:
			self.target_critic.load_state_dict(self.critic.state_dict())

		alpha = torch.load(path / 'alpha.pt', map_location=self.device)
		self.log_alpha.data.copy_(torch.as_tensor(alpha, dtype=torch.float32, device=self.device).reshape_as(self.log_alpha))
		return {"step": None, "metrics": {}, "config": {}}
