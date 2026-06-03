import numpy as np
from data.load import load_data
from rl_trading.configs.splits import WALK_FORWARD_FOLDS_2
from rl_trading.evaluation.baselines import (
    CashBaseline,
    EqualWeightBuyAndHold,
    MonthlyRebalanceEqualWeight,
    SPYBuyAndHold,
)
from rl_trading.evaluation.metrics import compute_portfolio_metrics

_, asset_returns, benchmark_returns = load_data()
fold = WALK_FORWARD_FOLDS_2[0]
val_start, val_end = fold["val"]
val_assets = asset_returns.loc[val_start:val_end].to_numpy(dtype=np.float32)
val_bench = benchmark_returns.loc[val_start:val_end].to_numpy(dtype=np.float32)
val_dates = asset_returns.loc[val_start:val_end].index

baselines = {
    "cash": CashBaseline(val_assets, val_bench, val_dates),
    "equal_weight_bh": EqualWeightBuyAndHold(val_assets, val_bench, val_dates),
    "equal_weight_monthly": MonthlyRebalanceEqualWeight(val_assets, val_bench, val_dates),
    "spy_bh": SPYBuyAndHold(val_assets, val_bench, val_dates),
}

for name, baseline in baselines.items():
    df = baseline.run()
    m = compute_portfolio_metrics(df)
    print(name, {
        "ann_return": round(m["annualized_return"], 4),
        "sharpe": round(m["sharpe_ratio"], 4),
        "max_dd": round(m["max_drawdown"], 4),
        "final_value": round(m["final_value"], 4),
    })
