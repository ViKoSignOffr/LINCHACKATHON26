from __future__ import annotations

import argparse
import sys
from bisect import bisect_right
from pathlib import Path

import pandas as pd

from trading_simulator import TradingSimulator
import custom_trading_simulator_3 as trend

ML_ROOT = Path(__file__).resolve().parent / "ml-trading-model"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.data.loader import load_data
from src.features.engineering import engineer_features
from src.model.predictor import make_predictions
from src.model.trainer import train_model
from src.strategy.cross_sectional import select_top_stocks


RAMP_STEP = 0.20
RAMP_YEARS = (3, 4, 5, 6, 7)
ML_TOP_N = 5
INITIAL_CASH = 100_000
BASE_CURRENCY = "Crncy_01"


def build_ramp_schedule(
    price_index: pd.DatetimeIndex,
    anchor_date: pd.Timestamp | None = None,
) -> list[tuple[pd.Timestamp, float]]:
    """Map each calendar anniversary threshold to the next available trading day."""
    if len(price_index) == 0:
        raise ValueError("Price data is empty.")

    if anchor_date is None:
        anchor_date = pd.Timestamp(price_index.min())
    else:
        anchor_date = pd.Timestamp(anchor_date)
        if anchor_date not in price_index:
            raise ValueError(f"Anchor date {anchor_date.date()} is not present in the price data.")

    schedule: list[tuple[pd.Timestamp, float]] = []
    for step_idx, years in enumerate(RAMP_YEARS, start=1):
        threshold_date = anchor_date + pd.DateOffset(years=years)
        trading_pos = price_index.searchsorted(threshold_date)
        if trading_pos >= len(price_index):
            break
        trading_date = pd.Timestamp(price_index[trading_pos])
        ml_share = min(step_idx * RAMP_STEP, 1.0)
        schedule.append((trading_date, ml_share))
    return schedule


def allocation_split(
    date: pd.Timestamp,
    ramp_schedule: list[tuple[pd.Timestamp, float]],
) -> tuple[float, float]:
    """Return (trend_share, ml_share) for the current trading date."""
    ml_share = 0.0
    current_date = pd.Timestamp(date)
    for threshold_date, threshold_share in ramp_schedule:
        if current_date >= threshold_date:
            ml_share = threshold_share
        else:
            break
    return 1.0 - ml_share, ml_share


def build_ml_rebalance_schedule(prices_csv: str, top_n: int = ML_TOP_N) -> pd.DataFrame:
    """Run the existing ML pipeline components and return top-N rebalance targets."""
    prices_wide = load_data(prices_csv)
    stock_prices = prices_wide[trend.STOCK_COLS].copy()

    features_df = engineer_features(stock_prices)
    trained_models = train_model(features_df)
    predictions_df = make_predictions(trained_models, features_df)
    rebalance_df = select_top_stocks(predictions_df, top_n=top_n)

    if rebalance_df.empty:
        raise RuntimeError("ML rebalance schedule is empty. Check ML dependencies and input data.")

    rebalance_df = rebalance_df.copy()
    rebalance_df["rebalance_date"] = pd.to_datetime(rebalance_df["rebalance_date"])
    return rebalance_df


def build_ml_rebalance_map(rebalance_df: pd.DataFrame) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, pd.Series]]:
    """Index the ML basket by rebalance date for fast daily lookup."""
    rebalance_map: dict[pd.Timestamp, pd.Series] = {}
    for rb_date, group in rebalance_df.groupby("rebalance_date"):
        weights = group.set_index("ticker")["weight"].astype(float)
        rebalance_map[pd.Timestamp(rb_date)] = weights

    rebalance_dates = sorted(rebalance_map)
    return rebalance_dates, rebalance_map


def latest_ml_targets(
    date: pd.Timestamp,
    rebalance_dates: list[pd.Timestamp],
    rebalance_map: dict[pd.Timestamp, pd.Series],
) -> pd.Series:
    """Return the latest active ML basket as of the provided date."""
    pos = bisect_right(rebalance_dates, pd.Timestamp(date)) - 1
    if pos < 0:
        return pd.Series(dtype=float)
    return rebalance_map[rebalance_dates[pos]]


def build_combined_weights(
    price_index: pd.Index,
    rebalance_dates: list[pd.Timestamp],
    rebalance_map: dict[pd.Timestamp, pd.Series],
    anchor_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Precompute daily combined target weights for the shared simulator."""
    combined_weight_df = pd.DataFrame(0.0, index=price_index, columns=trend.prices.columns)
    ramp_schedule = build_ramp_schedule(pd.DatetimeIndex(price_index), anchor_date=anchor_date)

    for date in price_index:
        trend_share, ml_share = allocation_split(date, ramp_schedule)

        scaled_trend_equity = trend.equity_target_weight_df.loc[date].mul(trend_share)
        scaled_trend_idx = trend.idx_target_weight_df.loc[date].mul(trend_share)
        scaled_trend_comm = trend.comm_target_weight_df.loc[date].mul(trend_share)

        ml_equity = pd.Series(0.0, index=trend.STOCK_COLS)
        if ml_share > 0:
            ml_targets = latest_ml_targets(date, rebalance_dates, rebalance_map)
            if not ml_targets.empty:
                ml_equity.loc[ml_targets.index] = ml_targets.values * ml_share

        total_equity = scaled_trend_equity.add(ml_equity, fill_value=0.0)
        fx_targets = trend.compute_fx_hedge_targets(total_equity.to_frame().T).iloc[0]

        combined_weight_df.loc[date, trend.STOCK_COLS] = total_equity
        combined_weight_df.loc[date, ["Idx_04"]] = scaled_trend_idx.values
        combined_weight_df.loc[date, ["Comm_01"]] = scaled_trend_comm.values
        combined_weight_df.loc[date, trend.FX_COLS] = fx_targets.values

    return combined_weight_df.fillna(0.0)


def build_strategy(target_weight_df: pd.DataFrame):
    """Create a TradingSimulator-compatible strategy from target weights."""

    def strategy_fn(row_pos, cash, portfolio, signal_prices, data):
        date = data.index[row_pos]
        if date not in target_weight_df.index:
            return []

        targets = target_weight_df.loc[date]
        holdings_value = sum(
            portfolio.get(ticker, 0) * signal_prices.get(ticker, 0.0)
            for ticker in target_weight_df.columns
        )
        capital_proxy = max(cash + holdings_value, 1.0)

        orders = []
        for ticker in target_weight_df.columns:
            weight = targets.get(ticker, 0.0)
            if pd.isna(weight):
                continue

            price = signal_prices.get(ticker)
            if pd.isna(price) or price == 0:
                continue

            target_notional = weight * capital_proxy
            target_shares = int(round(target_notional / price))
            delta = target_shares - portfolio.get(ticker, 0)

            if abs(delta) < trend.MIN_TRADE_SHARES:
                continue

            orders.append(("BUY" if delta > 0 else "SELL", ticker, abs(delta)))

        return orders

    return strategy_fn


def run_combined_simulation(
    prices_csv: str = "prices.csv",
    initial_cash: int = INITIAL_CASH,
    orders_file: str = "orders_combined.csv",
    portfolio_file: str = "portfolio_combined.csv",
    plot_file: str = "performance_plot_combined.png",
    anchor_date: str | None = None,
):
    prices_path = Path(prices_csv)
    prices_df = pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    anchor_timestamp = pd.Timestamp(anchor_date) if anchor_date is not None else pd.Timestamp(prices_df.index.min())

    rebalance_df = build_ml_rebalance_schedule(str(prices_path), top_n=ML_TOP_N)
    rebalance_dates, rebalance_map = build_ml_rebalance_map(rebalance_df)
    target_weight_df = build_combined_weights(
        prices_df.index,
        rebalance_dates,
        rebalance_map,
        anchor_date=anchor_timestamp,
    )

    simulator = TradingSimulator(
        assets=list(prices_df.columns),
        initial_cash=initial_cash,
        equity_currency_map=trend.equity_ccy,
        fx_pairs_map=trend.fx_pairs_map,
        base_currency=BASE_CURRENCY,
    )
    simulator.run(build_strategy(target_weight_df), prices_df, prices_df)
    simulator.save_results(orders_file=orders_file, portfolio_file=portfolio_file)
    simulator.plot_performance(prices_df, save_file=plot_file)

    return simulator, rebalance_df, target_weight_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices-csv", default="prices.csv")
    parser.add_argument("--initial-cash", type=int, default=INITIAL_CASH)
    parser.add_argument("--orders-file", default="orders_combined.csv")
    parser.add_argument("--portfolio-file", default="portfolio_combined.csv")
    parser.add_argument("--plot-file", default="performance_plot_combined.png")
    parser.add_argument("--anchor-date", default=None)
    args = parser.parse_args()

    run_combined_simulation(
        prices_csv=args.prices_csv,
        initial_cash=args.initial_cash,
        orders_file=args.orders_file,
        portfolio_file=args.portfolio_file,
        plot_file=args.plot_file,
        anchor_date=args.anchor_date,
    )


if __name__ == "__main__":
    main()
