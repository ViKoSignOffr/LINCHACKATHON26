import numpy as np
import pandas as pd
from trading_simulator import TradingSimulator

# ===== LOAD DATA =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]

print(f"Loaded {len(prices)} trading days, {len(STOCK_COLS)} assets")

# ===== STRATEGY =====
START_TARGET_SHARES = 150
END_TARGET_SHARES   = 25
NUM_LONGS           = 2
NUM_SHORTS          = 2


def compute_positions(stock_prices):
    stock_returns = stock_prices.pct_change()
    target_size_series = pd.Series(
        np.linspace(START_TARGET_SHARES, END_TARGET_SHARES, len(stock_prices)).round().astype(int),
        index=stock_prices.index,
    )

    target_positions = pd.DataFrame(0, index=stock_prices.index, columns=stock_prices.columns)

    for date in stock_prices.index:
        daily_returns = stock_returns.loc[date].dropna()
        if daily_returns.empty:
            continue

        target_shares = int(target_size_series.loc[date])
        longs = daily_returns.nsmallest(NUM_LONGS).index
        shorts = daily_returns.nlargest(NUM_SHORTS).index

        target_positions.loc[date, longs] = target_shares
        target_positions.loc[date, shorts] = -target_shares

    return target_positions


target_shares_df = compute_positions(prices[STOCK_COLS])


def strategy(row_pos, cash, portfolio, signal_prices, data):
    del cash, signal_prices

    orders = []
    date   = data.index[row_pos]

    if date not in target_shares_df.index:
        return orders

    targets = target_shares_df.loc[date]
    if targets.isna().all():
        return orders

    for ticker in STOCK_COLS:
        tgt   = targets.get(ticker, 0)
        if pd.isna(tgt):
            continue
        tgt   = int(round(tgt))
        delta = tgt - portfolio.get(ticker, 0)
        if delta != 0:
            orders.append(('BUY' if delta > 0 else 'SELL', ticker, abs(delta)))

    return orders


# ===== RUN SIMULATION =====
simulator = TradingSimulator(
    assets       = STOCK_COLS,
    initial_cash = 100_000,
)
simulator.run(strategy, prices[STOCK_COLS], prices)
simulator.save_results(
    orders_file    = "orders.csv",
    portfolio_file = "portfolio.csv",
)
simulator.plot_performance(prices[STOCK_COLS], save_file="performance_plot.png")
