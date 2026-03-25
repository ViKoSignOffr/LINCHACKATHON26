import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def select_top_stocks(
    predictions_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    At each rebalance date select the top-N stocks by predicted score,
    equal-weight them, and compute evaluation metrics.

    Parameters
    ----------
    predictions_df : output of make_predictions()
    top_n          : number of stocks to hold (default 5)

    Returns
    -------
    DataFrame with columns:
        rebalance_date, ticker, score, weight,
        basket_fwd_ret, universe_fwd_ret, spread,
        spearman_ic, hit_rate
    """
    results = []

    for rb_date, group in predictions_df.groupby('rebalance_date'):
        group = group.copy().sort_values('score', ascending=False)

        # ── Evaluation metrics (require target_fwd_21d) ──────────────────────
        has_target    = group['target_fwd_21d'].notna()
        scored_group  = group[has_target]

        universe_ret  = float(scored_group['target_fwd_21d'].mean()) \
                        if not scored_group.empty else np.nan

        # Spearman IC: correlation between predicted score and realised return
        ic, _         = spearmanr(scored_group['score'], scored_group['target_fwd_21d']) \
                        if len(scored_group) >= 3 else (np.nan, np.nan)

        # ── Select top-N ─────────────────────────────────────────────────────
        top           = group.head(top_n).copy()
        top['weight'] = 1.0 / len(top)

        basket_ret    = float(top['target_fwd_21d'].mean()) \
                        if top['target_fwd_21d'].notna().all() else np.nan
        spread        = basket_ret - universe_ret \
                        if not (np.isnan(basket_ret) or np.isnan(universe_ret)) else np.nan

        # Hit rate: fraction of top stocks that beat the universe median
        universe_med  = float(scored_group['target_fwd_21d'].median()) \
                        if not scored_group.empty else np.nan
        hit_rate      = float((top['target_fwd_21d'] > universe_med).mean()) \
                        if (not np.isnan(universe_med)
                            and top['target_fwd_21d'].notna().all()) else np.nan

        for _, stock_row in top.iterrows():
            results.append({
                'rebalance_date':  rb_date,
                'ticker':          stock_row['ticker'],
                'score':           stock_row['score'],
                'weight':          stock_row['weight'],
                'basket_fwd_ret':  basket_ret,
                'universe_fwd_ret':universe_ret,
                'spread':          spread,
                'spearman_ic':     ic,
                'hit_rate':        hit_rate,
            })

    return pd.DataFrame(results)


def rebalance_portfolio(predictions_df, rebalance_date):
    """Rebalance the portfolio based on predictions."""
    top_stocks = select_top_stocks(predictions_df)
    total_weight = len(top_stocks)
    top_stocks['weight'] = 1 / total_weight
    top_stocks['rebalance_date'] = rebalance_date
    return top_stocks[['rebalance_date', 'ticker', 'predicted_return', 'weight']]

def generate_rebalance_details(model, prices_df, rebalance_frequency='M'):
    """Generate rebalance details for the top stocks based on model predictions."""
    # Prepare the DataFrame for predictions
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    predictions_df = prices_df.groupby('ticker').apply(lambda x: model.predict(x[['close']])).reset_index()
    predictions_df.columns = ['ticker', 'predicted_return']

    # Generate rebalance dates
    rebalance_dates = prices_df['date'].dt.to_period(rebalance_frequency).drop_duplicates().dt.to_timestamp()

    rebalance_details = pd.DataFrame()
    for date in rebalance_dates:
        daily_predictions = predictions_df[predictions_df['date'] == date]
        rebalance = rebalance_portfolio(daily_predictions, date)
        rebalance_details = pd.concat([rebalance_details, rebalance], ignore_index=True)

    return rebalance_details.reset_index(drop=True)