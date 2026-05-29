# RL Trading Agent Plan

## Project Goal

Build a realistically completable reinforcement-learning trading project using
free daily data from `yfinance`.

The core project is a Soft Actor-Critic agent that manages a portfolio of sector
ETFs and is evaluated against simple, hard-to-fool baselines. The objective is
not to claim guaranteed profitability. The objective is to demonstrate a clean
sequential decision-making system with proper time-series methodology,
transaction costs, risk-aware evaluation, and honest out-of-sample testing.

The first complete version should trade:

- `XLK` - Technology
- `XLE` - Energy
- `XLF` - Financials
- `SPY` - benchmark only

After the 3-asset version is complete, expand to:

- `XLV` - Healthcare
- `XLI` - Industrials

Do not expand the asset universe until the 3-asset pipeline has passed all
checkpoints.

## Key Design Decisions

### Use Daily Data

Use daily bars for version 1.

The planned features, such as RSI-14, MACD 12/26, 5-day momentum, and 20-day
momentum, are naturally daily features. On minute data, these windows become too
short and noisy, and transaction costs dominate the signal. Daily data gives
enough history for meaningful train, validation, and test splits while keeping
the environment simple enough to finish.

Intraday data is a later extension. If this project moves intraday, use hourly
data before considering minute data.

### Reuse the Existing SAC Agent

Copy the SAC implementation from the visuomotor project and adapt it rather than
rewriting the algorithm.

Expected reusable pieces:

- Gaussian actor with tanh-squashed continuous actions
- twin Q critic
- target critic soft updates
- entropy coefficient logic
- replay buffer
- off-policy update loop
- model save/load pattern

Expected changes:

- remove any CNN/image encoder paths
- use flat vector observations
- set `obs_dim` from the trading environment
- set `action_dim = n_assets`
- log finance-specific metrics instead of robotics success metrics

Keep the SAC algorithm familiar. Most project risk should live in the trading
environment, leakage prevention, and evaluation, not in rewriting SAC.

### Constrain Portfolio Exposure

SAC can output one continuous action per asset, but raw independent actions in
`[-1, 1]` can accidentally create 300% gross exposure for 3 assets. Make the
portfolio constraint explicit.

Version 1 action interpretation:

```python
raw_action = np.tanh(actor_output)          # shape: (n_assets,)
target_weights = raw_action.copy()         # negative means short, positive means long
gross = np.abs(target_weights).sum()

if gross > max_gross_exposure:
    target_weights = target_weights / gross * max_gross_exposure
```

Use these starting constraints:

- `max_gross_exposure = 1.0`
- `max_weight_per_asset = 1.0`
- `transaction_cost_rate = 0.001`
- no interest on cash in version 1
- no borrow fees in version 1, but document this limitation if shorts are enabled

If shorting makes debugging harder, temporarily run a long-only version:

```python
target_weights = np.clip(raw_action, 0.0, 1.0)
target_weights = target_weights / max(target_weights.sum(), 1.0)
```

The long-short environment is the target. The long-only environment is an
acceptable debugging checkpoint.

## Data

### Source

Use `yfinance` for daily OHLCV data.

Initial symbols:

```python
ASSET_TICKERS = ["XLK", "XLE", "XLF"]
BENCHMARK_TICKER = "SPY"
START = "2010-01-01"
END = "2024-12-31"
```

Download adjusted daily data:

```python
import yfinance as yf

tickers = ASSET_TICKERS + [BENCHMARK_TICKER]
data = yf.download(tickers, start=START, end=END, auto_adjust=True)
```

Cache raw downloads locally so repeated training runs do not depend on network
availability.

### Chronological Splits

Never shuffle time-series data.

Use:

- training: `2010-01-01` to `2020-12-31`
- validation: `2021-01-01` to `2022-12-31`
- test: `2023-01-01` to `2024-12-31`

The test set is touched only once, after the model selection process is finished.
All debugging, hyperparameter changes, and checkpoint selection happen on the
training and validation sets.

### Leakage Rules

Every feature must satisfy these rules:

- features at day `t` use only information available before the trade for day `t`
- calculated close-based features are shifted by one day with `.shift(1)`
- rolling windows only look backward
- rows with warmup NaNs are dropped after all features are assembled
- raw `Close` price is not used directly as a model feature
- train/validation/test splits happen after feature construction, but no scaler
  is fit on validation or test data

### Feature Scaling

Fit any feature scaler on the training split only, then apply the frozen scaler
to validation and test.

Start simple:

- leave already bounded features as-is if they are naturally in a small range
- standardize return, volatility, volume, and correlation-derived features using
  training-split mean and standard deviation
- save the scaler parameters beside the model checkpoint
- never refit the scaler during validation or test evaluation

Checkpoint before moving on:

- a leakage test proves that changing future prices does not alter earlier
  feature rows
- feature rows have no NaNs after warmup removal
- all train/validation/test ranges are chronological and non-overlapping
- scaler parameters are learned from training data only

## Features

Calculate per-asset features for each tradable ETF.

Per asset:

- `returns_1d`
- `returns_5d`
- `returns_20d`
- `rsi_14`, normalized to `[-1, 1]`
- `macd_line`, normalized by price
- `macd_histogram`
- `bb_position`
- `volume_ratio`, clipped to a reasonable maximum such as `5`
- `atr_normalized`
- `price_position_20`

Cross-asset features:

- rolling 20-day correlations between asset returns
- for 3 assets: `corr_XLK_XLE`, `corr_XLK_XLF`, `corr_XLE_XLF`

Portfolio state features:

- current target/current portfolio weights
- per-asset unrealized return since latest position change, or a simpler
  position-level P&L proxy for version 1
- cash ratio
- current drawdown from episode peak

Expected observation dimension for 3 assets:

- 10 per-asset features x 3 assets = 30
- 3 correlation features = 3
- 8 portfolio-state features = 8
- total = 41

Checkpoint before moving on:

- feature matrix shape matches the expected observation dimension
- each feature column has a clear name and ticker association
- features are finite after warmup
- feature distributions look plausible from a simple summary or plot
- a single observation can be produced for any valid timestep

## Environment

Create a Gymnasium environment, `TradingEnv`, that receives precomputed features,
asset returns, and benchmark returns.

The environment should support:

- random 6-month training episodes
- deterministic full-period validation/test episodes
- continuous target-weight actions
- transaction costs based on changes in weights
- portfolio value tracking
- drawdown tracking
- info dictionaries with useful diagnostics

Recommended episode length:

```python
episode_length = 126  # about 6 trading months
```

### Step Timing

Use a simple and consistent execution assumption:

1. observation at step `t` contains shifted features known before trading day `t`
2. agent chooses target weights for day `t`
3. transaction cost is applied for changing from old weights to target weights
4. portfolio earns day `t` asset returns
5. benchmark earns day `t` SPY return
6. reward is calculated
7. environment advances to `t + 1`

Document this timing in the environment docstring.

### Reward

Start with excess return after transaction costs:

```python
portfolio_return = np.dot(target_weights, asset_returns_t)
turnover = np.abs(target_weights - old_weights).sum()
transaction_cost = turnover * transaction_cost_rate
net_portfolio_return = portfolio_return - transaction_cost
reward = net_portfolio_return - spy_return_t
```

Daily excess returns are small, often around `0.001` or less. SAC may learn more
reliably if the reward used for gradient updates is scaled while raw returns are
still logged:

```python
raw_reward = net_portfolio_return - spy_return_t
reward = raw_reward * reward_scale  # start with reward_scale = 100.0
```

Keep evaluation metrics based on raw, unscaled returns.

Keep the first reward simple. Add volatility penalties only after the baseline
environment is working and the effect can be measured.

Checkpoint before moving on:

- `reset()` returns the correct observation shape and finite values
- `step()` returns Gymnasium-compatible `(obs, reward, terminated, truncated, info)`
- zero action produces near-cash behavior
- constant long-only equal weights roughly match a hand-calculated return series
- transaction costs are charged only when positions change
- random-agent episodes run without NaNs or exploding portfolio value
- a random-agent equity curve can be plotted
- reward scaling changes learning rewards but not reported return metrics

## Agent Adaptation

Target class shapes:

```python
obs_dim = 41
action_dim = 3
hidden_dim = 256
```

Actor:

- input: observation vector
- output: mean and log standard deviation for each asset action
- sample with reparameterization
- apply tanh squashing and log-probability correction

Critic:

- input: concatenated observation and action
- output: scalar Q value
- use twin critics and the minimum target Q

Replay buffer:

- stores flat NumPy arrays or tensors
- no image storage
- no HER

Starting hyperparameters:

```python
HIDDEN_DIM = 256
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100_000
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
UPDATES_PER_STEP = 1
WARMUP_STEPS = 1_000
```

Checkpoint before moving on:

- actor forward pass returns action shape `(batch_size, 3)`
- critic forward pass returns Q shape `(batch_size, 1)` or `(batch_size,)`
- one sampled batch update runs without shape errors
- losses are finite
- actor parameters change after an update
- critic parameters change after an update
- target critic parameters move after a soft update

## Training

Training loop:

1. reset environment to a random training window
2. use random actions during warmup
3. use SAC policy actions after warmup
4. store transitions in replay
5. update once per environment step after enough samples exist
6. log episode and update metrics
7. periodically run deterministic validation episodes
8. save checkpoints by validation Sharpe, not training reward

Log at minimum:

- episode reward
- episode portfolio return
- episode benchmark return
- validation Sharpe
- validation max drawdown
- average gross exposure
- average turnover
- transaction costs paid
- actor loss
- critic loss
- entropy / alpha

Checkpoint before moving on:

- 10 random episodes run cleanly
- 10 policy episodes run cleanly before learning starts
- 500 training episodes complete without NaNs
- replay buffer size grows as expected
- losses remain finite
- validation evaluation is deterministic for a fixed checkpoint
- saved checkpoint can be reloaded and evaluated

Do not tune aggressively until these mechanical checks pass.

## Evaluation

Use validation for development and model selection. Use test only once for final
reporting.

Metrics:

- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio, optional
- max drawdown
- Calmar ratio
- win rate
- average turnover
- average gross exposure
- total transaction costs

Baselines:

1. SPY buy and hold
2. equal-weight buy and hold across tradable ETFs
3. equal-weight monthly rebalance
4. random-action agent with the same action constraints
5. cash / flat portfolio

The final model should be compared against all baselines on the same date range,
with the same transaction-cost assumptions where applicable.

Checkpoint before final test:

- validation results beat at least the random-action baseline
- evaluation code can reproduce identical metrics for the same checkpoint
- all baselines are implemented and plotted
- no hyperparameters are changed based on test results

Final test checkpoint:

- run the chosen checkpoint once on `2023-01-01` to `2024-12-31`
- compute all metrics
- plot equity curve against SPY and equal-weight baselines
- plot position weights over time
- write down limitations honestly

## Suggested Repository Structure

Use a `src` layout so imports and tests stay clean:

```text
RL-trading/
├── PLAN.md
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/
│   └── processed/
├── plots/
├── src/
│   └── rl_trading/
│       ├── __init__.py
│       ├── data/
│       │   ├── download.py
│       │   └── features.py
│       ├── envs/
│       │   └── trading_env.py
│       ├── agents/
│       │   ├── actor.py
│       │   ├── critic.py
│       │   ├── sac.py
│       │   └── buffer.py
│       ├── training/
│       │   └── train.py
│       └── evaluation/
│           ├── baselines.py
│           ├── metrics.py
│           └── evaluate.py
└── tests/
    ├── test_features_no_leakage.py
    ├── test_env_accounting.py
    └── test_sac_shapes.py
```

## Build Stages

### Stage 0 - Project Skeleton

Goal: make the repo runnable before implementing trading logic.

Tasks:

- add dependencies for `yfinance`, `gymnasium`, `torch`, `matplotlib`, and tests
- create the `src/rl_trading` package
- create empty data, environment, agent, training, and evaluation modules
- add a basic test command

Checkpoint:

- `uv run pytest` runs
- imports work from the package
- no trading code has to be correct yet

### Stage 1 - Data Download and Cache

Goal: produce reliable local OHLCV data.

Tasks:

- download XLK, XLE, XLF, and SPY daily data
- cache raw data under `data/raw/`
- normalize yfinance column format into a predictable schema
- verify date coverage
- handle missing rows explicitly

Checkpoint:

- local cached data can be loaded without network access
- all symbols share the same trading calendar after alignment
- adjusted OHLCV data is finite for the usable date range

### Stage 2 - Feature Pipeline

Goal: build leak-free observations.

Tasks:

- implement per-asset features
- implement rolling correlations
- shift close-based features by one day
- split chronologically
- add a no-leakage test
- add feature summary diagnostics

Checkpoint:

- feature matrix has expected shape
- no feature uses future data
- train/validation/test splits are clean
- one observation row can be traced back to its source dates

### Stage 3 - Trading Environment

Goal: make the MDP mechanically correct.

Tasks:

- implement `TradingEnv`
- implement target-weight action constraints
- implement transaction costs
- implement portfolio value, cash ratio, and drawdown
- implement random-window training resets
- implement deterministic validation/test resets
- add accounting tests

Checkpoint:

- random actions run for 10 episodes
- equal-weight policy matches independent baseline calculations
- transaction costs behave correctly
- observations, rewards, and info values stay finite

### Stage 4 - Baselines and Metrics

Goal: know what the agent has to beat before training it.

Tasks:

- implement SPY buy-and-hold baseline
- implement equal-weight buy-and-hold baseline
- implement equal-weight monthly rebalance baseline
- implement random-action baseline
- implement metrics and equity plotting

Checkpoint:

- all baselines run on validation and test periods
- metrics are reproducible
- plots are generated
- baseline numbers are saved before SAC training begins

### Stage 5 - SAC Port

Goal: copy the SAC agent and make it work with flat trading observations.

Tasks:

- copy actor, critic, SAC update logic, and replay buffer
- remove image/CNN assumptions
- wire in `obs_dim` and `action_dim`
- add shape tests
- add one-update smoke test
- add checkpoint save/load

Checkpoint:

- actor, critic, and update tests pass
- losses are finite on a fake batch
- parameters update as expected
- saved checkpoint reloads

### Stage 6 - First Training Run

Goal: prove the full system can train end to end.

Tasks:

- connect SAC to `TradingEnv`
- run warmup with random actions
- train for a small number of episodes
- log training metrics
- run periodic validation
- save checkpoints

Checkpoint:

- 500 episodes complete without NaNs
- validation evaluation works from a saved checkpoint
- average gross exposure and turnover are plausible
- policy does not collapse immediately to always-flat or always-max-leverage

### Stage 7 - Debugging and Tuning

Goal: improve only after the system is measurable.

Tasks:

- tune one hyperparameter at a time
- inspect position time series
- inspect turnover and transaction costs
- adjust transaction cost only if there is a clear diagnosis
- compare long-only and long-short variants if needed

Checkpoint:

- best checkpoint is selected by validation Sharpe or validation Calmar
- selected checkpoint beats random-action validation baseline
- behavior is explainable from plots, not just one scalar metric

### Stage 8 - Final Test Evaluation

Goal: produce the honest final result.

Tasks:

- freeze code and hyperparameters
- run the selected checkpoint once on the test set
- compute all metrics
- compare against every baseline
- generate final plots

Checkpoint:

- test metrics are saved
- equity curve plot is saved
- position heatmap or weight plot is saved
- limitations are documented
- no further tuning is done from test feedback

### Stage 9 - README and Writeup

Goal: make the project understandable to someone reviewing it.

Tasks:

- explain the data source and split
- explain observation, action, reward, and constraints
- explain leakage prevention
- report validation and test metrics
- compare to baselines
- discuss limitations
- include setup and run commands

Checkpoint:

- a reviewer can reproduce the main result
- the methodology is clear even if performance is modest
- claims are supported by plots and metrics

## Completion Standard

This project is complete when:

- data can be downloaded and cached
- features are leak-free and tested
- the trading environment passes accounting checks
- baselines are implemented before agent tuning
- SAC trains end to end
- the final checkpoint is selected using validation data only
- the test set is evaluated once
- results are compared against meaningful baselines
- limitations are documented honestly

The strongest version of this project is not the one with the highest backtest
return. It is the one with the cleanest methodology.

## Resume Framing

Possible one-line description:

> Trained a continuous Soft Actor-Critic agent for multi-asset sector ETF
> portfolio management using daily market data, evaluated against SPY,
> equal-weight, monthly-rebalanced, and random-action baselines with Sharpe,
> Calmar, and maximum drawdown metrics.
