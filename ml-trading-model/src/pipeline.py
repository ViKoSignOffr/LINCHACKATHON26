import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from src.data.loader import load_data
from src.features.engineering import engineer_features
from src.model.trainer import train_model
from src.model.predictor import make_predictions
from src.strategy.cross_sectional import select_top_stocks
from trading_simulator import TradingSimulator

# ── FX Configuration ─────────────────────────────────────────────────────────
# Map each equity ticker to its denominated currency
EQUITY_CURRENCY_MAP = {
    # e.g. 'AAPL': 'USD', 'VOW3': 'EUR'
    # Fill in your tickers here
}

# Map each FX ticker in your price CSV to (base_ccy, quote_ccy)
FX_PAIRS_MAP = {
    # e.g. 'FX_EURUSD': ('EUR', 'USD')
    # Fill in your FX pairs here
}

BASE_CURRENCY = 'Crncy_01'   # match your simulator default

# Fraction of FX exposure to hedge (1.0 = full hedge, 0.5 = half hedge)
HEDGE_RATIO = 1.0


def build_strategy(rebalance_df: pd.DataFrame, fx_tickers: list[str]):
    """
    Return a strategy_fn compatible with TradingSimulator.run().

    On each rebalance:
      1. Rebalances equity basket (top-5, equal weight)
      2. Sells old FX hedges
      3. Opens new FX hedges proportional to each currency's equity exposure
    """
    rebalance_df = rebalance_df.copy()
    rebalance_df['rebalance_date'] = pd.to_datetime(rebalance_df['rebalance_date'])

    rebalance_map: dict[pd.Timestamp, dict[str, float]] = {}
    for rb_date, grp in rebalance_df.groupby('rebalance_date'):
        rebalance_map[pd.Timestamp(rb_date)] = dict(zip(grp['ticker'], grp['weight']))

    rebalance_dates     = sorted(rebalance_map.keys())
    current_targets:    dict[str, float] = {}

    def strategy_fn(row_pos, cash, portfolio, prices, data):
        nonlocal current_targets

        prices_df: pd.DataFrame = data['prices_df']
        today = pd.Timestamp(prices_df.index[row_pos])

        past = [d for d in rebalance_dates if d <= today]
        if not past:
            return []

        new_targets = rebalance_map[max(past)]
        if new_targets == current_targets:
            return []

        current_targets = new_targets
        orders = []

        total_value = cash + sum(
            portfolio.get(t, 0) * prices.get(t, 0) for t in portfolio
        )

        # ── 1. Close old FX hedges ────────────────────────────────────────
        for fx_ticker in fx_tickers:
            held = portfolio.get(fx_ticker, 0)
            if held > 0:
                orders.append(('SELL', fx_ticker, held))
            elif held < 0:
                orders.append(('BUY',  fx_ticker, abs(held)))

        # ── 2. Sell equities no longer in basket ─────────────────────────
        for ticker, shares in portfolio.items():
            if ticker in fx_tickers:
                continue
            if shares > 0 and ticker not in new_targets:
                orders.append(('SELL', ticker, shares))

        # ── 3. Buy / rebalance equity basket ─────────────────────────────
        currency_exposure: dict[str, float] = {}   # ccy → base-ccy value

        for ticker, weight in new_targets.items():
            target_value  = total_value * weight
            current_price = prices.get(ticker, 0)
            if current_price <= 0:
                continue

            target_shares  = int(target_value / current_price)
            current_shares = portfolio.get(ticker, 0)
            delta          = target_shares - current_shares

            if delta > 0:
                orders.append(('BUY',  ticker, delta))
            elif delta < 0:
                orders.append(('SELL', ticker, -delta))

            # Track currency exposure from this position
            ccy = EQUITY_CURRENCY_MAP.get(ticker)
            if ccy and ccy != BASE_CURRENCY:
                currency_exposure[ccy] = (
                    currency_exposure.get(ccy, 0.0) + target_shares * current_price
                )

        # ── 4. Open new FX hedges ─────────────────────────────────────────
        # For each foreign-currency exposure, sell the FX pair to hedge
        # (selling FX(base/quote) offsets long foreign equity exposure)
        for fx_ticker, (base_ccy, quote_ccy) in FX_PAIRS_MAP.items():
            fx_price = prices.get(fx_ticker, 0)
            if fx_price <= 0:
                continue

            # Exposure in foreign currency units
            exposure = currency_exposure.get(base_ccy, 0.0)
            if exposure == 0:
                exposure = currency_exposure.get(quote_ccy, 0.0)
            if exposure == 0:
                continue

            hedge_notional = exposure * HEDGE_RATIO
            hedge_shares   = int(hedge_notional / fx_price)

            if hedge_shares > 0:
                # Sell FX to create offsetting short position
                orders.append(('SELL', fx_ticker, hedge_shares))

        return orders

    return strategy_fn


def main(prices_csv: str = 'data/prices.csv'):
    prices_wide = load_data(prices_csv)
    features_df = engineer_features(prices_wide)

    trained_models = train_model(features_df)
    predictions_df = make_predictions(trained_models, features_df)

    rebalance_df = select_top_stocks(predictions_df, top_n=5)
    rebalance_df.to_csv('data/rebalance_schedule.csv', index=False)
    print(rebalance_df.to_string(index=False))

    # ── Tickers: equity basket + FX hedges ───────────────────────────────────
    equity_tickers = rebalance_df['ticker'].unique().tolist()
    fx_tickers     = list(FX_PAIRS_MAP.keys())
    all_tickers    = equity_tickers + fx_tickers

    sim = TradingSimulator(
        assets              = all_tickers,
        initial_cash        = 100_000,
        equity_currency_map = EQUITY_CURRENCY_MAP,
        fx_pairs_map        = FX_PAIRS_MAP,
        base_currency       = BASE_CURRENCY,
    )

    sim_prices = prices_wide[
        [t for t in all_tickers if t in prices_wide.columns]
    ].dropna(how='all')

    strategy = build_strategy(rebalance_df, fx_tickers)
    sim.run(strategy, sim_prices, data={'prices_df': sim_prices})

    sim.save_results(
        orders_file    = 'data/orders.csv',
        portfolio_file = 'data/portfolio.csv',
    )
    sim.plot_performance(sim_prices, save_file='data/performance_plot.png')

    return rebalance_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices-csv', default='data/prices.csv')
    parser.add_argument('--plot-only',  action='store_true')
    parser.add_argument('--save-plot',  default='data/performance_plot.png')
    args = parser.parse_args()

    if args.plot_only:
        from src.pipeline import plot_only
        plot_only(prices_csv=args.prices_csv, save_file=args.save_plot)
    else:
        main(prices_csv=args.prices_csv)