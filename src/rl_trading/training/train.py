import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Import project configs
from rl_trading.configs.stock_universe import ASSET_TICKERS
from rl_trading.configs.splits import WALK_FORWARD_FOLDS, WALK_FORWARD_FOLDS_2
from rl_trading.configs.data_classes import ModelParameters, OffPolicyConfig

# Import data, env, agent, and trainer classes
from data.load import load_data
from rl_trading.envs.trading_env import TradingEnv
from rl_trading.agents.sac import SACAgent
from rl_trading.training.buffer import ReplayBuffer
from rl_trading.training.trainer import OffPolicyTrainer
from rl_trading.training.loggers import TensorBoardLogger


def run_training(fold_idx: int = 3, total_steps: int = 100_000):
	"""
	Orchestrates loading, scaling, environment setup, and SAC training on a selected fold.
	"""
	# 1. Load the pre-processed aligned features and returns
	features, asset_returns, benchmark_returns = load_data()

	# 2. Get dates for the selected walk-forward fold
	fold = WALK_FORWARD_FOLDS_2[fold_idx]
	train_dates = fold["train"]
	val_dates = fold["val"]

	print(f"--- Training on {fold['name']} ---")
	print(f"Train Period: {train_dates[0]} to {train_dates[1]}")
	print(f"Val Period:   {val_dates[0]} to {val_dates[1]}")

	# 3. Slice Dataframes Chronologically
	train_feat_raw = features.loc[train_dates[0]:train_dates[1]]
	train_assets = asset_returns.loc[train_dates[0]:train_dates[1]]
	train_bench = benchmark_returns.loc[train_dates[0]:train_dates[1]]

	val_feat_raw = features.loc[val_dates[0]:val_dates[1]]
	val_assets = asset_returns.loc[val_dates[0]:val_dates[1]]
	val_bench = benchmark_returns.loc[val_dates[0]:val_dates[1]]

	# 4. Fit Scaler on TRAIN set only (Crucial for Leakage Prevention)
	# Calculate statistics from train set
	train_mean = train_feat_raw.mean(axis=0)
	train_std = train_feat_raw.std(axis=0) + 1e-8

	# Apply standardization to both Train and Val
	train_feat_scaled = ((train_feat_raw - train_mean) / train_std).to_numpy(dtype=np.float32)
	val_feat_scaled = ((val_feat_raw - train_mean) / train_std).to_numpy(dtype=np.float32)

	# Convert returns to raw numpy arrays for the Gymnasium environment
	train_assets_np = train_assets.to_numpy(dtype=np.float32)
	train_bench_np = train_bench.to_numpy(dtype=np.float32)

	val_assets_np = val_assets.to_numpy(dtype=np.float32)
	val_bench_np = val_bench.to_numpy(dtype=np.float32)

	# 5. Create Environments
	# Train mode uses a shorter random episode_length to force diversity and exploration
	train_env = TradingEnv(
		features=train_feat_scaled,
		asset_returns=train_assets_np,
		benchmark_returns=train_bench_np,
		dates=train_feat_raw.index,
		episode_length=252,
		transaction_cost_rate=0.001,
		max_gross_exposure=1.0,
		allow_short=True,
		reward_scale=100.0,
		reward_mode="risk_adjusted",
		downside_penalty=2.0,
		drawdown_penalty=0.1,
		turnover_penalty=0.0005,
		mode="train"
	)

	# Eval mode covers the full validation timeframe deterministically
	val_env = TradingEnv(
		features=val_feat_scaled,
		asset_returns=val_assets_np,
		benchmark_returns=val_bench_np,
		dates=val_feat_raw.index,
		episode_length=len(val_feat_raw),
		transaction_cost_rate=0.001,
		max_gross_exposure=1.0,
		allow_short=True,
		reward_scale=100.0,
		reward_mode="risk_adjusted",
		downside_penalty=2.0,
		drawdown_penalty=0.1,
		turnover_penalty=0.0005,
		mode="eval"
	)

	# 6. Instantiate Replay Buffer & Agent
	n_assets = len(ASSET_TICKERS)
	obs_dim = train_env.observation_space.shape[0]

	# Leverage MPS automatically on Apple Silicon
	device = "mps" if torch.backends.mps.is_available() else "cpu"
	print(f"Using device: {device}")

	agent_params = ModelParameters(
		hidden_dims=(256, 256),
		actor_lr=3e-4,
		critic_lr=3e-4,
		alpha_lr=3e-4,
		gamma=0.99,
		tau=0.005,
		target_entropy=-2.0
	)

	agent = SACAgent(
		obs_size=obs_dim,
		action_size=n_assets,
		config=agent_params,
		device=device
	)

	buffer = ReplayBuffer(
		buffer_size=250_000,
		obs_space=train_env.observation_space,
		act_space=train_env.action_space,
		device=device
	)

	# 7. Configure Loggers & Trainer
	log_dir = f"runs/{fold['name']}_sac"
	logger = TensorBoardLogger(log_dir=log_dir)

	checkpoint_dir = Path(f"checkpoints/{fold['name']}")
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	# Save the feature scaler parameters along with the checkpoints
	# so we can use the exact same mean and std during out-of-sample test evaluations later
	np.save(Path(checkpoint_dir) / "feature_scaler_mean.npy", train_mean.to_numpy())
	np.save(Path(checkpoint_dir) / "feature_scaler_std.npy", train_std.to_numpy())

	trainer_config = OffPolicyConfig(
		warmup_steps=2_000,
		update_after=2_000,
		update_interval=1,
		update_every=1,
		batch_size=256,
		eval_interval=5_000,
		checkpoint_dir=checkpoint_dir,
		best_metric="validation/sharpe",
		save_interval=20_000
	)

	trainer = OffPolicyTrainer(
		env=train_env,
		eval_env=val_env,
		agent=agent,
		buffer=buffer,
		config=trainer_config,
		logger=logger
	)

	# 8. Start training!
	print(f"Starting training for {total_steps} steps...")
	trainer.train(total_steps=total_steps)

	logger.close()
	print("Training finished!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--fold", type=int, default=3, help="Fold index to train (0 to 3)")
	parser.add_argument("--steps", type=int, default=100_000, help="Total environment steps to train")
	args = parser.parse_args()

	run_training(fold_idx=args.fold, total_steps=args.steps)
