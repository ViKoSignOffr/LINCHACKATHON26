import numpy as np
import pandas as pd
from pathlib import Path
from trading_simulator import TradingSimulator

# ===== LOAD DATA =====
# Load the full anonymized market panel once at module scope so every helper
# works from the same source data and we only pay the I/O cost one time.
prices = pd.read_csv(Path("prices.csv"), index_col="Date", parse_dates=True)

# Split the universe into asset-class buckets. Later helpers use these groups
# to assign different roles to equities, indices, commodities, and FX.
STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
IDX_COLS   = [c for c in prices.columns if c.startswith("Idx_")]
COMM_COLS  = [c for c in prices.columns if c.startswith("Comm_")]
FX_COLS    = [c for c in prices.columns if c.startswith("FX_")]

print(f"Loaded {len(prices)} trading days, {len(prices.columns)} assets")

# ===== ASSET METADATA =====
# Equities carry explicit currency exposure, so we need this mapping to know
# which stocks create foreign-currency risk inside the portfolio.
equity_ccy = {
    'Stock_01': 'Crncy_03', 'Stock_02': 'Crncy_04', 'Stock_03': 'Crncy_04',
    'Stock_04': 'Crncy_02', 'Stock_05': 'Crncy_03', 'Stock_06': 'Crncy_02',
    'Stock_07': 'Crncy_03', 'Stock_08': 'Crncy_02', 'Stock_09': 'Crncy_04',
    'Stock_10': 'Crncy_03', 'Stock_11': 'Crncy_01', 'Stock_12': 'Crncy_04',
    'Stock_13': 'Crncy_01', 'Stock_14': 'Crncy_01', 'Stock_15': 'Crncy_01',
}

# These FX pairs are the instruments we can use to translate foreign equity
# exposure back toward the base currency. The simulator also uses them when it
# calculates portfolio value and hedge metrics.
fx_pairs_map = {
    'FX_01': ('Crncy_02', 'Crncy_01'),
    'FX_02': ('Crncy_04', 'Crncy_02'),
    'FX_03': ('Crncy_04', 'Crncy_03'),
    'FX_04': ('Crncy_02', 'Crncy_03'),
    'FX_05': ('Crncy_01', 'Crncy_03'),
    'FX_06': ('Crncy_04', 'Crncy_01'),
}

# ===== STRATEGY =====
# Trend lookbacks: fast versus slow moving averages define the market direction.
FAST_WINDOW      = 40
SLOW_WINDOW      = 120
# Volatility lookback: recent realized volatility is used to scale position size.
VOL_WINDOW       = 60
# Scaling and caps: stop any single raw score or final position from dominating.
MAX_RAW_STRENGTH = 3.0
MAX_ASSET_WEIGHT = 0.12
# Trading friction control: ignore tiny target changes that would just create churn.
MIN_TRADE_SHARES = 5
# FX hedge strength: hedge part of the foreign equity exposure, but not necessarily all of it.
FX_HEDGE_RATIO   = 0.75

# The portfolio is intentionally equity-led. Indices, commodities, and FX are
# supporting sleeves rather than equal peers in the risk budget.
CLASS_CAPS = {
    'Stock': 0.70,
    'Idx': 0.15,
    'Comm': 0.05,
    'FX': 0.10,
}

# Direct hedge map: each foreign currency is matched to the cleanest FX pair
# versus the base currency. Crncy_01 does not appear because it is already the
# base currency and therefore does not need hedging.
FX_HEDGE_MAP = {
    'Crncy_02': ('FX_01', -1.0),
    'Crncy_03': ('FX_05',  1.0),
    'Crncy_04': ('FX_06', -1.0),
}


def get_asset_class(ticker):
    """Map a ticker to the risk bucket used for portfolio construction."""
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
    """Compute the core directional signal from fast versus slow moving averages."""
    fast_ma = asset_prices.rolling(FAST_WINDOW, min_periods=FAST_WINDOW).mean()
    slow_ma = asset_prices.rolling(SLOW_WINDOW, min_periods=SLOW_WINDOW).mean()
    relative_trend = fast_ma.div(slow_ma).sub(1.0)

    return np.sign(relative_trend).fillna(0.0)


def compute_rolling_vol(asset_prices):
    """Estimate recent risk so more volatile assets receive smaller positions."""
    daily_returns = asset_prices.pct_change()
    rolling_vol = daily_returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

    return rolling_vol.replace(0, np.nan)


def compute_equity_targets(raw_scores, idx04_signal):
    """Build the main long-only stock sleeve and scale it by the broad market regime."""
    # Make the strategy benchmark-aware by letting stocks drive returns and
    # by reducing, rather than inverting, equity risk in weak index regimes.
    stock_scores = raw_scores[STOCK_COLS].clip(lower=0.0)
    gross = stock_scores.abs().sum(axis=1).replace(0, np.nan)
    targets = stock_scores.div(gross, axis=0).mul(CLASS_CAPS['Stock']).clip(upper=MAX_ASSET_WEIGHT).fillna(0.0)

    regime_scale = pd.Series(np.where(idx04_signal > 0, 1.0, 0.5), index=targets.index)
    regime_scale = regime_scale.where(idx04_signal.notna(), 0.0)

    return targets.mul(regime_scale, axis=0)


def compute_idx04_targets(raw_scores):
    """Use Idx_04 as a small trend overlay and as the benchmark-aware index sleeve."""
    idx04_scores = raw_scores[['Idx_04']]
    gross = idx04_scores.abs().sum(axis=1).replace(0, np.nan)
    targets = idx04_scores.div(gross, axis=0).mul(CLASS_CAPS['Idx'])

    return targets.clip(-MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT).fillna(0.0)


def compute_commodity_targets(raw_scores):
    """Keep commodities simple: Comm_01 is only a small diversifying overlay."""
    comm_scores = raw_scores[['Comm_01']]
    gross = comm_scores.abs().sum(axis=1).replace(0, np.nan)
    targets = comm_scores.div(gross, axis=0).mul(CLASS_CAPS['Comm'])

    return targets.clip(-MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT).fillna(0.0)


def compute_fx_hedge_targets(equity_targets):
    """Convert foreign equity exposure into a direct FX hedge sleeve."""
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
    """Merge the sleeve-specific targets into one final portfolio target table."""
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    targets.update(equity_targets)
    targets.update(idx_targets)
    targets.update(comm_targets)
    targets.update(fx_targets)

    return targets.fillna(0.0)


# Precompute the entire pipeline once so the daily strategy function only has
# to read already-prepared targets:
# 1. compute directional trend
# 2. scale by volatility
# 3. zero out invalid or pre-warmup periods
# 4. build each sleeve separately
# 5. combine sleeves into one portfolio target table
trend_signal_df = compute_trend_signal(prices)
rolling_vol_df  = compute_rolling_vol(prices)

raw_position_df = trend_signal_df.div(rolling_vol_df)
raw_position_df = raw_position_df.replace([np.inf, -np.inf], np.nan).clip(-MAX_RAW_STRENGTH, MAX_RAW_STRENGTH)
raw_position_df = raw_position_df.where(rolling_vol_df.notna(), 0.0).fillna(0.0)
raw_position_df.iloc[:SLOW_WINDOW - 1] = 0.0

# Idx_04 is the market regime input for the stock sleeve, so we keep a clean
# one-column series for it after the warmup/validity filters.
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
    # The simulator calls this once per day. Our job here is to translate the
    # precomputed target weights into actual buy/sell orders for that date.
    orders = []
    date   = data.index[row_pos]

    # Do not trade until the slow lookback is available, otherwise the trend
    # signal would be based on incomplete history.
    if row_pos < SLOW_WINDOW - 1:
        return orders
    if date not in target_weight_df.index:
        return orders

    targets = target_weight_df.loc[date]
    if targets.abs().sum() == 0:
        return orders

    # Convert abstract portfolio weights into share counts using an estimate of
    # current portfolio capital. This keeps the target sizing economically meaningful.
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

        # Skip tiny changes so the portfolio does not rebalance on meaningless noise.
        if abs(delta) < MIN_TRADE_SHARES:
            continue

        # Orders are always generated as deltas versus current holdings.
        orders.append(('BUY' if delta > 0 else 'SELL', ticker, abs(delta)))

    return orders


def run_simulation(
    initial_cash=100_000,
    orders_file="orders_trend_multi_asset.csv",
    portfolio_file="portfolio_trend_multi_asset.csv",
    plot_file="performance_plot_trend_multi_asset.png",
):
    # Pass the currency and FX maps into the simulator so it can value foreign
    # holdings correctly and report FX hedge metrics.
    simulator = TradingSimulator(
        assets=list(prices.columns),
        initial_cash=initial_cash,
        equity_currency_map=equity_ccy,
        fx_pairs_map=fx_pairs_map,
    )
    simulator.run(strategy, prices, prices)

    # Save a separate order blotter and portfolio history for this strategy version.
    simulator.save_results(
        orders_file=orders_file,
        portfolio_file=portfolio_file,
    )

    # Save the performance dashboard so the strategy can be inspected visually.
    simulator.plot_performance(prices, save_file=plot_file)
    return simulator


if __name__ == "__main__":
    run_simulation()
