import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from rl_trading.configs.data_classes import ModelParameters, OffPolicyConfig
from rl_trading.agents.sac import SACAgent
from rl_trading.envs.trading_env import TradingEnv
from rl_trading.training.buffer import ReplayBuffer
from rl_trading.training.trainer import OffPolicyTrainer
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.load import load_data


class DummyLogger:
	def log_scalar(self, key: str, value: float, step: int):
		pass

	def log_dict(self, metrics: dict[str, float], step: int):
		pass

	def close(self):
		pass


def test_end_to_end_integration(tmp_path):
	features, asset_returns, benchmark_returns = load_data()
	
	assert len(features) > 200, "Features dataset too small for integration test."
	assert len(asset_returns) == len(features)
	assert len(benchmark_returns) == len(features)

	train_size = 150
	val_size = 50
	
	train_features = features.iloc[:train_size]
	train_asset_returns = asset_returns.iloc[:train_size]
	train_benchmark_returns = benchmark_returns.iloc[:train_size]
	train_dates = features.index[:train_size]
	
	val_features = features.iloc[train_size:train_size+val_size]
	val_asset_returns = asset_returns.iloc[train_size:train_size+val_size]
	val_benchmark_returns = benchmark_returns.iloc[train_size:train_size+val_size]
	val_dates = features.index[train_size:train_size+val_size]

	mean = train_features.mean(axis=0)
	std = train_features.std(axis=0) + 1e-8
	
	train_features_scaled = ((train_features - mean) / std).to_numpy(dtype=np.float32)
	val_features_scaled = ((val_features - mean) / std).to_numpy(dtype=np.float32)
	
	train_asset_returns_np = train_asset_returns.to_numpy(dtype=np.float32)
	train_benchmark_returns_np = train_benchmark_returns.to_numpy(dtype=np.float32)
	
	val_asset_returns_np = val_asset_returns.to_numpy(dtype=np.float32)
	val_benchmark_returns_np = val_benchmark_returns.to_numpy(dtype=np.float32)

	train_env = TradingEnv(
		features=train_features_scaled,
		asset_returns=train_asset_returns_np,
		benchmark_returns=train_benchmark_returns_np,
		dates=train_dates,
		episode_length=40,
		transaction_cost_rate=0.001,
		max_gross_exposure=1.0,
		max_weight_per_asset=1.0,
		allow_short=True,
		reward_scale=10.0,
		mode='train',
	)
	
	val_env = TradingEnv(
		features=val_features_scaled,
		asset_returns=val_asset_returns_np,
		benchmark_returns=val_benchmark_returns_np,
		dates=val_dates,
		episode_length=val_size,
		transaction_cost_rate=0.001,
		max_gross_exposure=1.0,
		max_weight_per_asset=1.0,
		allow_short=True,
		reward_scale=10.0,
		mode='eval',
	)

	n_assets = train_asset_returns_np.shape[1]
	obs_dim = train_env.observation_space.shape[0]
	
	agent_config = ModelParameters(
		hidden_dims=(32, 32),
		actor_lr=3e-4,
		critic_lr=3e-4,
	)
	agent = SACAgent(
		obs_size=obs_dim,
		action_size=n_assets,
		config=agent_config,
		device="cpu",
	)
	
	buffer = ReplayBuffer(
		buffer_size=100,
		obs_space=SimpleNamespace(shape=(obs_dim,)),
		act_space=SimpleNamespace(shape=(n_assets,)),
		device="cpu",
	)

	trainer_config = OffPolicyConfig(
		warmup_steps=5,
		update_after=5,
		update_interval=1,
		update_every=1,
		batch_size=4,
		eval_interval=10,
		checkpoint_dir=str(tmp_path),
		best_metric="validation/sharpe",
	)
	
	trainer = OffPolicyTrainer(
		env=train_env,
		eval_env=val_env,
		agent=agent,
		buffer=buffer,
		config=trainer_config,
		logger=DummyLogger(),
	)

	trainer.train(total_steps=15)

	assert len(buffer) == 15

	eval_metrics = trainer.evaluate()
	
	expected_keys = {
		"validation/sharpe",
		"validation/max_drawdown",
		"validation/portfolio_return",
		"validation/average_turnover",
		"validation/average_gross_exposure",
		"validation/transaction_costs",
	}
	
	assert expected_keys.issubset(eval_metrics.keys())
	for key in expected_keys:
		val = eval_metrics[key]
		assert np.isfinite(val), f"Metric {key} is not finite: {val}"
