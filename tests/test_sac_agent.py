from types import SimpleNamespace

import numpy as np
import pytest

from rl_trading.agents.sac import SACAgent
from rl_trading.training.buffer import ReplayBuffer
from rl_trading.configs.data_classes import ModelParameters, OffPolicyConfig
from rl_trading.training.trainer import OffPolicyTrainer


class DummyLogger:
	def log_scalar(self, *_args, **_kwargs):
		pass

	def log_dict(self, *_args, **_kwargs):
		pass


def _small_config() -> ModelParameters:
	return ModelParameters(hidden_dims=(32, 32))


def _fill_buffer(buffer: ReplayBuffer, obs_size: int, action_size: int, n: int = 64):
	rng = np.random.default_rng(123)
	for _ in range(n):
		obs = rng.normal(size=obs_size).astype(np.float32)
		action = rng.uniform(-1, 1, size=action_size).astype(np.float32)
		reward = np.float32(rng.normal() * 0.01)
		next_obs = rng.normal(size=obs_size).astype(np.float32)
		buffer.add(obs, action, reward, next_obs, False, False)


def test_sac_update_runs_with_finite_losses():
	obs_size = 41
	action_size = 3
	agent = SACAgent(obs_size, action_size, _small_config(), device="cpu")
	buffer = ReplayBuffer(
		buffer_size=128,
		obs_space=SimpleNamespace(shape=(obs_size,)),
		act_space=SimpleNamespace(shape=(action_size,)),
		device="cpu",
	)
	_fill_buffer(buffer, obs_size, action_size)

	metrics = agent.update(buffer.sample(batch_size=32))

	assert set(metrics) == {"critic_loss", "actor_loss", "alpha_loss", "alpha"}
	assert all(np.isfinite(value) for value in metrics.values())
	assert metrics["alpha"] > 0


def test_checkpoint_round_trip_restores_policy_and_metadata(tmp_path):
	obs_size = 41
	action_size = 3
	config = _small_config()
	agent = SACAgent(obs_size, action_size, config, device="cpu")
	obs = np.linspace(-1, 1, obs_size, dtype=np.float32)
	expected_action = agent.select_action(obs, deterministic=True)

	agent.save(tmp_path, step=123, metrics={"validation/sharpe": 1.25})

	loaded_agent = SACAgent(obs_size, action_size, config, device="cpu")
	metadata = loaded_agent.load(tmp_path)
	loaded_action = loaded_agent.select_action(obs, deterministic=True)

	np.testing.assert_allclose(loaded_action, expected_action)
	assert metadata["step"] == 123
	assert metadata["metrics"] == {"validation/sharpe": 1.25}



def test_trainer_saves_best_checkpoint_by_configured_metric(tmp_path):
	agent = SACAgent(41, 3, _small_config(), device="cpu")
	config = OffPolicyConfig(checkpoint_dir=str(tmp_path), best_metric="validation/sharpe")
	trainer = OffPolicyTrainer(None, None, agent, None, config, DummyLogger())

	trainer._save_best_if_needed({"validation/sharpe": 1.0}, step=10)
	trainer._save_best_if_needed({"validation/sharpe": 0.5}, step=20)

	metadata = agent.load_best(tmp_path)

	assert metadata["step"] == 10
	assert metadata["metrics"] == {"validation/sharpe": 1.0}
