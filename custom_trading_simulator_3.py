import numpy as np
import pandas as pd
from pathlib import Path
from trading_simulator import TradingSimulator

# ===== LOAD DATA =====
prices = pd.read_csv(Path("prices.csv"), index_col="Date", parse_dates=True)

STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
IDX_COLS   = [c for c in prices.columns if c.startswith("Idx_")]
COMM_COLS  = [c for c in prices.columns if c.startswith("Comm_")]
FX_COLS    = [c for c in prices.columns if c.startswith("FX_")]

print(f"Loaded {len(prices)} trading days, {len(prices.columns)} assets")

# ===== ASSET METADATA =====
equity_ccy = {
    'Stock_01': 'Crncy_03', 'Stock_02': 'Crncy_04', 'Stock_03': 'Crncy_04',
    'Stock_04': 'Crncy_02', 'Stock_05': 'Crncy_03', 'Stock_06': 'Crncy_02',
    'Stock_07': 'Crncy_03', 'Stock_08': 'Crncy_02', 'Stock_09': 'Crncy_04',
    'Stock_10': 'Crncy_03', 'Stock_11': 'Crncy_01', 'Stock_12': 'Crncy_04',
    'Stock_13': 'Crncy_01', 'Stock_14': 'Crncy_01', 'Stock_15': 'Crncy_01',
}

fx_pairs_map = {
    'FX_01': ('Crncy_02', 'Crncy_01'),
    'FX_02': ('Crncy_04', 'Crncy_02'),
    'FX_03': ('Crncy_04', 'Crncy_03'),
    'FX_04': ('Crncy_02', 'Crncy_03'),
    'FX_05': ('Crncy_01', 'Crncy_03'),
    'FX_06': ('Crncy_04', 'Crncy_01'),
}

# ===== STRATEGY =====
FAST_WINDOW      = 40
SLOW_WINDOW      = 120
VOL_WINDOW       = 60
MAX_RAW_STRENGTH = 3.0
MAX_ASSET_WEIGHT = 0.12
MIN_TRADE_SHARES = 5
FX_HEDGE_RATIO   = 0.75

CLASS_CAPS = {
    'Stock': 0.70,
    'Idx': 0.15,
    'Comm': 0.05,
    'FX': 0.10,
}

FX_HEDGE_MAP = {
    'Crncy_02': ('FX_01', -1.0),
    'Crncy_03': ('FX_05',  1.0),
    'Crncy_04': ('FX_06', -1.0),
}


def get_asset_class(ticker):
    if ticker.startswith("Stock_"):
        return "Stock"
    if ticker.startswith("Idx_"):
        return "Idx"
    if ticker.startswith("Comm_"):
        return "Comm"
    if ticker.startswith("FX_"):
        return "FX"
    raise ValueError(f"Unknown asset class for {ticker}")


def compute_trend_signal(asset_prices):
    fast_ma = asset_prices.rolling(FAST_WINDOW, min_periods=FAST_WINDOW).mean()
    slow_ma = asset_prices.rolling(SLOW_WINDOW, min_periods=SLOW_WINDOW).mean()
    relative_trend = fast_ma.div(slow_ma).sub(1.0)

    return np.sign(relative_trend).fillna(0.0)


def compute_rolling_vol(asset_prices):
    daily_returns = asset_prices.pct_change()
    rolling_vol = daily_returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

    return rolling_vol.replace(0, np.nan)


def compute_equity_targets(raw_scores, idx04_signal):
    # Make the strategy benchmark-aware by letting stocks drive returns and
    # by reducing, rather than inverting, equity risk in weak index regimes.
    stock_scores = raw_scores[STOCK_COLS].clip(lower=0.0)
    gross = stock_scores.abs().sum(axis=1).replace(0, np.nan)
    targets = stock_scores.div(gross, axis=0).mul(CLASS_CAPS['Stock']).clip(upper=MAX_ASSET_WEIGHT).fillna(0.0)

    regime_scale = pd.Series(np.where(idx04_signal > 0, 1.0, 0.5), index=targets.index)
    regime_scale = regime_scale.where(idx04_signal.notna(), 0.0)

    return targets.mul(regime_scale, axis=0)


def compute_idx04_targets(raw_scores):
    idx04_scores = raw_scores[['Idx_04']]
    gross = idx04_scores.abs().sum(axis=1).replace(0, np.nan)
    targets = idx04_scores.div(gross, axis=0).mul(CLASS_CAPS['Idx'])

    return targets.clip(-MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT).fillna(0.0)


def compute_commodity_targets(raw_scores):
    comm_scores = raw_scores[['Comm_01']]
    gross = comm_scores.abs().sum(axis=1).replace(0, np.nan)
    targets = comm_scores.div(gross, axis=0).mul(CLASS_CAPS['Comm'])

    return targets.clip(-MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT).fillna(0.0)


def compute_fx_hedge_targets(equity_targets):
    # FX is now a hedge sleeve: it offsets the foreign-currency part of the
    # equity book instead of acting as a separate confirmation overlay.
    hedge_targets = pd.DataFrame(0.0, index=equity_targets.index, columns=FX_COLS)

    for ccy, (fx_ticker, direction) in FX_HEDGE_MAP.items():
        currency_weight = pd.Series(0.0, index=equity_targets.index)
        currency_stocks = [ticker for ticker in STOCK_COLS if equity_ccy.get(ticker) == ccy]
        if currency_stocks:
            currency_weight = equity_targets[currency_stocks].sum(axis=1)

        hedge_targets[fx_ticker] = direction * FX_HEDGE_RATIO * currency_weight

    return hedge_targets.fillna(0.0)


def combine_targets(equity_targets, idx_targets, comm_targets, fx_targets):
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    targets.update(equity_targets)
    targets.update(idx_targets)
    targets.update(comm_targets)
    targets.update(fx_targets)

    return targets.fillna(0.0)


trend_signal_df = compute_trend_signal(prices)
rolling_vol_df  = compute_rolling_vol(prices)

raw_position_df = trend_signal_df.div(rolling_vol_df)
raw_position_df = raw_position_df.replace([np.inf, -np.inf], np.nan).clip(-MAX_RAW_STRENGTH, MAX_RAW_STRENGTH)
raw_position_df = raw_position_df.where(rolling_vol_df.notna(), 0.0).fillna(0.0)
raw_position_df.iloc[:SLOW_WINDOW - 1] = 0.0

idx04_signal_series = trend_signal_df['Idx_04'].where(rolling_vol_df['Idx_04'].notna(), 0.0)

equity_target_weight_df = compute_equity_targets(raw_position_df, idx04_signal_series)
idx_target_weight_df    = compute_idx04_targets(raw_position_df)
comm_target_weight_df   = compute_commodity_targets(raw_position_df)
fx_target_weight_df     = compute_fx_hedge_targets(equity_target_weight_df)
target_weight_df        = combine_targets(
    equity_target_weight_df,
    idx_target_weight_df,
    comm_target_weight_df,
    fx_target_weight_df,
)


def strategy(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    date   = data.index[row_pos]

    if row_pos < SLOW_WINDOW - 1:
        return orders
    if date not in target_weight_df.index:
        return orders

    targets = target_weight_df.loc[date]
    if targets.abs().sum() == 0:
        return orders

    holdings_value = sum(portfolio.get(ticker, 0) * signal_prices.get(ticker, 0.0) for ticker in prices.columns)
    capital_proxy = max(cash + holdings_value, 1.0)

    for ticker in prices.columns:
        weight = targets.get(ticker, 0.0)
        if pd.isna(weight):
            continue

        price = signal_prices.get(ticker)
        if pd.isna(price) or price == 0:
            continue

        target_notional = weight * capital_proxy
        tgt   = int(round(target_notional / price))
        delta = tgt - portfolio.get(ticker, 0)

        if abs(delta) < MIN_TRADE_SHARES:
            continue
        orders.append(('BUY' if delta > 0 else 'SELL', ticker, abs(delta)))

    return orders


# ===== RUN SIMULATION =====
simulator = TradingSimulator(
    assets              = list(prices.columns),
    initial_cash        = 100_000,
    equity_currency_map = equity_ccy,
    fx_pairs_map        = fx_pairs_map,
)
simulator.run(strategy, prices, prices)
simulator.save_results(
    orders_file    = "orders_trend_multi_asset.csv",
    portfolio_file = "portfolio_trend_multi_asset.csv",
)
simulator.plot_performance(prices, save_file="performance_plot_trend_multi_asset.png")
