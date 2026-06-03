import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': []}

    PORTFOLIO_STATE_KEYS = [
        "cash_ratio",  # 1 - sum(weights)
        "gross_exposure",  # sum(|weights|)
        "net_exposure",  # sum(weights)
        "unrealized_pnl",  # portfolio_value - 1.0 (episode return so far)
        "drawdown_from_peak",  # 1 - portfolio_value / peak
        "episode_progress",  # steps_taken / episode_length, in [0, 1]
        "last_turnover",  # turnover from the previous step
        "last_transaction_cost",
        "return_ewma",
        "return_vol_ewma",
        "downside_vol_ewma",
    ]


    def __init__(self,
    features: np.ndarray,
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    dates: np.ndarray | pd.DatetimeIndex,
    *,
    episode_length: int = 126,
    transaction_cost_rate: float = 0.001,
    max_gross_exposure: float = 1.0,
    max_weight_per_asset: float = 1.0,
    allow_short: bool = False,
    reward_scale: float = 100.0,
    reward_mode: str = "risk_adjusted",
    benchmark_weight: float = 0.0,
    downside_penalty: float = 2.0,
    drawdown_penalty: float = 0.1,
    turnover_penalty: float = 0.0005,
    reward_ewma_window: int = 20,
    mode: str = 'train',
    seed: int | None = None,
    ):
        super().__init__()
        assert mode in ['train', 'eval']
        if reward_mode not in {"net_return", "excess_return", "risk_adjusted"}:
            raise ValueError(
                "reward_mode must be one of {'net_return', 'excess_return', 'risk_adjusted'}"
            )

        self.features = np.asarray(features, dtype=np.float32)
        self.asset_returns = np.asarray(asset_returns, dtype=np.float32)
        self.benchmark_returns = np.asarray(benchmark_returns, dtype=np.float32)
        self.dates = dates

        self._base_episode_length = episode_length
        self.episode_length = episode_length
        self.transaction_cost_rate = transaction_cost_rate
        self.max_gross_exposure = max_gross_exposure
        self.max_weight_per_asset = max_weight_per_asset
        self.allow_short = allow_short
        self.reward_scale = reward_scale
        self.reward_mode = reward_mode
        self.benchmark_weight = benchmark_weight
        self.downside_penalty = downside_penalty
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.reward_ewma_alpha = 2.0 / (float(reward_ewma_window) + 1.0)
        self.mode = mode

        self.n_assets = asset_returns.shape[1]
        self.T = len(self.features)
        n_portfolio_states = len(self.PORTFOLIO_STATE_KEYS)
        obs_dim = features.shape[1] + self.n_assets + n_portfolio_states
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._t0: int = 0
        self._t: int = 0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value: float = 1.0
        self.peak: float = 1.0
        self.last_turnover: float = 0.0
        self.last_transaction_cost: float = 0.0
        self.return_ewma: float = 0.0
        self.return_sq_ewma: float = 0.0
        self.downside_sq_ewma: float = 0.0



        if seed is not None:
            self.reset(seed=seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self.mode == 'train':
            max_start = self.T - self._base_episode_length
            if max_start < 0:
                raise ValueError(
                    f"Not enough data for episode_length={self._base_episode_length}. "
                    f"T={self.T}"
                )
            self._t0 = int(self.np_random.integers(0, max_start))
            self.episode_length = self._base_episode_length
        else:
            self._t0 = 0
            self.episode_length = len(self.features)

        self._t = self._t0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        self.peak = 1.0
        self.last_turnover = 0.0
        self.last_transaction_cost = 0.0
        self.return_ewma = 0.0
        self.return_sq_ewma = 0.0
        self.downside_sq_ewma = 0.0


        return self._observe(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        target = self._project(action)

        turnover = float(np.abs(target - self.weights).sum())
        transaction_cost = turnover * self.transaction_cost_rate

        portfolio_return_t = float(np.dot(target, self.asset_returns[self._t]))
        net_return_t = portfolio_return_t - transaction_cost

        benchmark_return_t = float(self.benchmark_returns[self._t])
        old_drawdown = float(1.0 - self.portfolio_value / self.peak)

        new_portfolio_value = self.portfolio_value * (1.0 + net_return_t)
        new_peak = max(self.peak, new_portfolio_value)
        new_drawdown = float(1.0 - new_portfolio_value / new_peak)

        reward_components = self._reward_components(
            net_return=net_return_t,
            benchmark_return=benchmark_return_t,
            turnover=turnover,
            old_drawdown=old_drawdown,
            new_drawdown=new_drawdown,
        )
        raw_reward = reward_components["raw_reward"]
        reward = float(raw_reward * self.reward_scale)

        self.weights = target.copy()
        self.portfolio_value = new_portfolio_value
        self.peak = new_peak
        self.last_turnover = turnover
        self.last_transaction_cost = transaction_cost
        self._update_return_stats(net_return_t)

        current_t = self._t
        self._t += 1
        steps_taken = self._t - self._t0
        truncated = steps_taken >= self.episode_length or self._t >= self.T
        terminated = False

        info = {
            "date": self.dates[current_t],
            "weights": self.weights.copy(),
            "cash": float(1.0 - self.weights.sum()),
            "gross_exposure": float(np.abs(self.weights).sum()),
            "net_exposure": float(self.weights.sum()),
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "portfolio_return": portfolio_return_t,
            "net_return": net_return_t,
            "benchmark_return": benchmark_return_t,
            **reward_components,
            "portfolio_value": self.portfolio_value,
            "drawdown": new_drawdown,
            "return_ewma": self.return_ewma,
            "return_vol_ewma": self._return_vol_ewma(),
            "downside_vol_ewma": self._downside_vol_ewma(),
        }

        return self._observe(), reward, terminated, truncated, info

    def _reward_components(
        self,
        *,
        net_return: float,
        benchmark_return: float,
        turnover: float,
        old_drawdown: float,
        new_drawdown: float,
    ) -> dict[str, float]:
        log_return = float(np.log1p(max(net_return, -0.999999)))
        excess_log_return = log_return - self.benchmark_weight * benchmark_return

        if self.reward_mode == "net_return":
            raw_reward = net_return
            downside_cost = 0.0
            drawdown_cost = 0.0
            turnover_cost = 0.0
        elif self.reward_mode == "excess_return":
            raw_reward = net_return - benchmark_return
            downside_cost = 0.0
            drawdown_cost = 0.0
            turnover_cost = 0.0
        else:
            downside = min(excess_log_return, 0.0)
            drawdown_increase = max(new_drawdown - old_drawdown, 0.0)
            downside_cost = self.downside_penalty * downside * downside
            drawdown_cost = self.drawdown_penalty * drawdown_increase
            turnover_cost = self.turnover_penalty * turnover
            raw_reward = excess_log_return - downside_cost - drawdown_cost - turnover_cost

        return {
            "raw_reward": float(raw_reward),
            "reward_log_return": log_return,
            "reward_excess_log_return": excess_log_return,
            "reward_downside_penalty": float(downside_cost),
            "reward_drawdown_penalty": float(drawdown_cost),
            "reward_turnover_penalty": float(turnover_cost),
        }

    def _update_return_stats(self, net_return: float) -> None:
        alpha = self.reward_ewma_alpha
        downside = min(net_return, 0.0)
        self.return_ewma = (1.0 - alpha) * self.return_ewma + alpha * net_return
        self.return_sq_ewma = (1.0 - alpha) * self.return_sq_ewma + alpha * net_return * net_return
        self.downside_sq_ewma = (1.0 - alpha) * self.downside_sq_ewma + alpha * downside * downside

    def _return_vol_ewma(self) -> float:
        variance = max(self.return_sq_ewma - self.return_ewma * self.return_ewma, 0.0)
        return float(np.sqrt(variance))

    def _downside_vol_ewma(self) -> float:
        return float(np.sqrt(max(self.downside_sq_ewma, 0.0)))

    def _observe(self):
        t = min(self._t, self.T - 1)
        market_features = self.features[t]
        portfolio_state = self._portfolio_state()

        return np.concatenate([market_features, self.weights, portfolio_state], dtype=np.float32)

    def _portfolio_state(self):
        steps_taken = self._t - self._t0

        return np.array(
            [
                float(1.0 - self.weights.sum()),                    # cash_ratio
                float(np.abs(self.weights).sum()),                  # gross_exposure
                float(self.weights.sum()),                          # net_exposure
                float(self.portfolio_value - 1.0),                  # unrealized_pnl
                float(1.0 - self.portfolio_value / self.peak),      # drawdown_from_peak
                float(steps_taken) / max(float(self.episode_length), 1),    # episode_progress
                self.last_turnover,                                 # last_turnover
                self.last_transaction_cost,                         # last_transaction_cost
                self.return_ewma,                                    # return_ewma
                self._return_vol_ewma(),                             # return_vol_ewma
                self._downside_vol_ewma(),                           # downside_vol_ewma
            ],
            dtype=np.float32,
        )

    def _info(self):

        return {
            "date": self.dates[self._t],
            "weights": self.weights.copy(),
            "cash": float(1.0 - self.weights.sum()),
            "gross_exposure": float(np.abs(self.weights).sum()),
            "net_exposure": float(self.weights.sum()),
            "portfolio_value": self.portfolio_value,
            "drawdown": float(1.0 - self.portfolio_value / self.peak),
            "return_ewma": self.return_ewma,
            "return_vol_ewma": self._return_vol_ewma(),
            "downside_vol_ewma": self._downside_vol_ewma(),
        }

    def _project(self, raw_action: np.ndarray) -> np.ndarray:
        weights = np.clip(raw_action, -1.0, 1.0)

        if not self.allow_short:
            weights = np.clip(weights, 0.0, 1.0)

        weights = np.clip(weights, -self.max_weight_per_asset, self.max_weight_per_asset)

        gross = float(np.abs(weights).sum())
        if gross > self.max_gross_exposure and gross > 0.0:
            weights = weights / gross * self.max_gross_exposure

        return weights.astype(np.float32)

    def close(self):
        pass

    def render(self, mode='human'):
        pass
