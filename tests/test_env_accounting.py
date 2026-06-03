import numpy as np
import pandas as pd
import pytest

from rl_trading.envs.trading_env import TradingEnv


N_ASSETS = 3
N_FEATURES = 33
T = 400
EPISODE_LENGTH = 126
OBS_DIM = N_FEATURES + N_ASSETS + len(TradingEnv.PORTFOLIO_STATE_KEYS)


@pytest.fixture
def env_arrays():
	rng = np.random.default_rng(0)
	features = rng.standard_normal((T, N_FEATURES)).astype(np.float32)
	asset_returns = rng.normal(0.0005, 0.01, size=(T, N_ASSETS)).astype(np.float32)
	benchmark_returns = rng.normal(0.0004, 0.009, size=T).astype(np.float32)
	dates = pd.date_range("2015-01-01", periods=T, freq="B")
	return features, asset_returns, benchmark_returns, dates


def make_env(env_arrays, **kwargs):
	features, asset_returns, benchmark_returns, dates = env_arrays
	defaults = dict(
		episode_length=EPISODE_LENGTH,
		transaction_cost_rate=0.001,
		max_gross_exposure=1.0,
		max_weight_per_asset=1.0,
		allow_short=False,
		reward_scale=100.0,
		mode='train',
		seed=123,
	)
	defaults.update(kwargs)
	return TradingEnv(features, asset_returns, benchmark_returns, dates, **defaults)


def test_reset_shapes_and_finite(env_arrays):
	env = make_env(env_arrays)
	obs, info = env.reset(seed=0)
	assert obs.shape == (OBS_DIM,)
	assert np.all(np.isfinite(obs))
	assert env.observation_space.contains(obs)
	assert info["portfolio_value"] == 1.0


def test_step_returns_gymnasium_tuple(env_arrays):
	env = make_env(env_arrays)
	env.reset(seed=0)
	action = np.zeros(N_ASSETS, dtype=np.float32)
	out = env.step(action)
	assert len(out) == 5
	obs, reward, terminated, truncated, info = out
	assert obs.shape == (OBS_DIM,)
	assert isinstance(reward, float)
	assert isinstance(terminated, bool)
	assert isinstance(truncated, bool)


def test_zero_action_is_cash_and_reward_is_zero(env_arrays):
	env = make_env(env_arrays)
	env.reset(seed=0)
	t = env._t
	_, reward, _, _, info = env.step(np.zeros(N_ASSETS, dtype=np.float32))

	assert info["turnover"] == 0.0
	assert info["transaction_cost"] == 0.0
	assert info["portfolio_return"] == 0.0
	expected_raw = 0.0
	np.testing.assert_allclose(info["raw_reward"], expected_raw, rtol=1e-6, atol=1e-8)
	np.testing.assert_allclose(reward, expected_raw * env.reward_scale, rtol=1e-6, atol=1e-6)
	assert info["cash"] == pytest.approx(1.0)


def test_risk_adjusted_reward_exposes_penalty_components(env_arrays):
	env = make_env(
		env_arrays,
		transaction_cost_rate=0.0,
		reward_scale=1.0,
		reward_mode="risk_adjusted",
		downside_penalty=2.0,
		drawdown_penalty=0.1,
		turnover_penalty=0.0,
	)
	env.reset(seed=0)
	t = env._t
	env.asset_returns[t] = np.array([-0.02, 0.0, 0.0], dtype=np.float32)

	_, reward, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

	assert info["reward_log_return"] < 0.0
	assert info["reward_downside_penalty"] > 0.0
	assert info["reward_drawdown_penalty"] > 0.0
	assert info["return_vol_ewma"] >= 0.0
	assert info["downside_vol_ewma"] > 0.0
	np.testing.assert_allclose(reward, info["raw_reward"], rtol=1e-6, atol=1e-8)


def test_equal_weight_matches_manual_calculation(env_arrays):
	env = make_env(env_arrays, transaction_cost_rate=0.0)
	env.reset(seed=0)
	t0 = env._t0
	w = np.full(N_ASSETS, 1.0 / N_ASSETS, dtype=np.float32)

	manual_value = 1.0
	for i in range(20):
		t = env._t
		_, _, _, _, info = env.step(w)
		manual_ret = float(np.dot(w, env.asset_returns[t]))
		manual_value *= 1.0 + manual_ret
		np.testing.assert_allclose(info["portfolio_return"], manual_ret, rtol=1e-5, atol=1e-7)
		np.testing.assert_allclose(info["portfolio_value"], manual_value, rtol=1e-5, atol=1e-7)
	assert env._t == t0 + 20


def test_transaction_cost_zero_when_weights_unchanged(env_arrays):
	env = make_env(env_arrays)
	env.reset(seed=0)
	w = np.array([0.3, 0.3, 0.3], dtype=np.float32)
	env.step(w)
	_, _, _, _, info = env.step(w)
	assert info["turnover"] == pytest.approx(0.0, abs=1e-7)
	assert info["transaction_cost"] == pytest.approx(0.0, abs=1e-7)


def test_transaction_cost_charged_when_weights_change(env_arrays):
	env = make_env(env_arrays, transaction_cost_rate=0.01)
	env.reset(seed=0)
	w1 = np.array([0.5, 0.0, 0.0], dtype=np.float32)
	w2 = np.array([0.0, 0.5, 0.0], dtype=np.float32)
	env.step(w1)
	_, _, _, _, info = env.step(w2)
	# turnover = |0-0.5| + |0.5-0| + |0-0| = 1.0
	assert info["turnover"] == pytest.approx(1.0, abs=1e-6)
	assert info["transaction_cost"] == pytest.approx(0.01, abs=1e-7)


def test_random_agent_rollout_is_finite(env_arrays):
	env = make_env(env_arrays)
	obs, _ = env.reset(seed=42)
	rng = np.random.default_rng(42)
	terminated = truncated = False
	while not (terminated or truncated):
		action = rng.uniform(-1, 1, size=N_ASSETS).astype(np.float32)
		obs, reward, terminated, truncated, info = env.step(action)
		assert np.all(np.isfinite(obs))
		assert np.isfinite(reward)
		assert np.isfinite(info["portfolio_value"])
		assert info["portfolio_value"] > 0.0
		assert 0.0 <= info["drawdown"] <= 1.0
		assert info["gross_exposure"] <= env.max_gross_exposure + 1e-6


def test_train_episode_length_is_exact(env_arrays):
	env = make_env(env_arrays)
	env.reset(seed=7)
	steps = 0
	terminated = truncated = False
	while not (terminated or truncated):
		_, _, terminated, truncated, _ = env.step(np.zeros(N_ASSETS, dtype=np.float32))
		steps += 1
	assert steps == EPISODE_LENGTH


def test_eval_mode_covers_full_span_deterministically(env_arrays):
	env = make_env(env_arrays, mode='eval')
	env.reset(seed=0)
	assert env._t0 == 0

	steps = 0
	terminated = truncated = False
	while not (terminated or truncated):
		_, _, terminated, truncated, _ = env.step(np.zeros(N_ASSETS, dtype=np.float32))
		steps += 1
	assert steps == T

	# Determinism: same seed → same trajectory
	env2 = make_env(env_arrays, mode='eval')
	env2.reset(seed=0)
	rng = np.random.default_rng(0)
	actions = [rng.uniform(0, 1, size=N_ASSETS).astype(np.float32) for _ in range(10)]

	env3 = make_env(env_arrays, mode='eval')
	env3.reset(seed=0)
	for a in actions:
		o2, r2, _, _, _ = env2.step(a)
		o3, r3, _, _, _ = env3.step(a)
		np.testing.assert_array_equal(o2, o3)
		assert r2 == r3


def test_train_reset_is_random_and_seeded(env_arrays):
	env = make_env(env_arrays)
	env.reset(seed=1)
	t0_a = env._t0
	env.reset(seed=2)
	t0_b = env._t0
	env.reset(seed=1)
	t0_c = env._t0
	assert t0_a == t0_c
	# With T=400 and ep_len=126, max_start=274 → different seeds should usually give different starts
	assert t0_a != t0_b


def test_long_only_projects_negative_actions_to_zero(env_arrays):
	env = make_env(env_arrays, allow_short=False)
	env.reset(seed=0)
	_, _, _, _, info = env.step(np.array([-1.0, -0.5, 0.4], dtype=np.float32))
	w = info["weights"]
	assert (w >= 0.0).all()
	assert w[0] == 0.0 and w[1] == 0.0
	assert w[2] == pytest.approx(0.4, abs=1e-6)


def test_gross_exposure_constraint(env_arrays):
	env = make_env(env_arrays, max_gross_exposure=1.0, allow_short=True)
	env.reset(seed=0)
	_, _, _, _, info = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
	assert info["gross_exposure"] == pytest.approx(1.0, abs=1e-6)
