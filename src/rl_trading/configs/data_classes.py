from dataclasses import dataclass
import torch

@dataclass
class Transition:
	"""One batch of environment transitions sampled from the replay buffer.

	Attributes:
		obs: Observation before taking the action.
		act: Action applied in the environment from `obs`.
		rew: Reward returned by the environment for the transition.
		next_obs: Observation returned after applying `act`.
		terminated: Whether the episode ended because the task reached a terminal state.
		truncated: Whether the episode ended because an external limit was reached.
	"""
	obs: torch.Tensor
	act: torch.Tensor
	rew: torch.Tensor
	next_obs: torch.Tensor
	terminated: torch.Tensor
	truncated: torch.Tensor


@dataclass
class ModelParameters:
	"""Network and optimizer settings for the SAC agent.

	Attributes:
		actor_lr: Learning rate for the policy network optimizer.
		critic_lr: Learning rate for the Q-function critic optimizer.
		alpha_lr: Learning rate for the entropy-temperature optimizer.
		gamma: Discount factor used in Bellman targets.
		tau: Soft-update rate for the target critic.
		initial_alpha: Initial entropy-temperature value.
		hidden_dims: Hidden-layer widths for actor and critic MLPs.
		log_std_min: Lower clamp for actor log standard deviation.
		log_std_max: Upper clamp for actor log standard deviation.
		target_entropy: Entropy target. If `None`, SAC uses `-action_dim`.
	"""
	actor_lr: float = 3e-4
	critic_lr: float = 3e-4
	alpha_lr: float = 3e-4
	gamma: float = 0.99
	tau: float = 0.005
	initial_alpha: float = 0.2
	hidden_dims: tuple[int, ...] = (256, 256)
	log_std_min: float = -20.0
	log_std_max: float = 2.0
	target_entropy: float | None = None

@dataclass
class OffPolicyConfig:
	"""Training-loop settings for the off-policy SAC trainer.

	Attributes:
		warmup_steps: Number of initial environment steps that use random actions
			before the agent policy is used.
		update_interval: Number of environment steps between update phases.
		update_every: Number of gradient updates to run each time an update phase starts.
		eval_interval: Number of environment steps between evaluation runs.
		batch_size: Number of replay-buffer transitions sampled for each gradient update.
		eval_trials: Number of episodes used to estimate evaluation success rate.
		update_after: Minimum environment step before training updates are allowed.
		save_interval: Number of environment steps between checkpoint saves, or `None`
			to disable periodic checkpointing.
		checkpoint_dir: Directory used for periodic and best-model checkpoints.
		best_metric: Evaluation metric key used to select the best checkpoint.
	"""
	warmup_steps: int = 1_000
	update_interval: int = 1
	update_every: int = 1
	eval_interval: int | None = None
	batch_size: int = 256
	eval_trials: int = 1
	update_after: int = 1_000
	save_interval: int|None = None
	checkpoint_dir: str = "./checkpoints"
	best_metric: str = "validation/sharpe"
