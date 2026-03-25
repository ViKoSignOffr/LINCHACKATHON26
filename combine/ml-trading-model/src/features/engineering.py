import pandas as pd
import numpy as np

WINDOWS = [5, 21, 63, 126, 252]
VOL_WINDOWS = [21, 63]
MA_WINDOWS = [21, 63, 126]
MIN_HISTORY = 252 + 1  # need at least 252 days to compute all features


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-format feature DataFrame.

    Columns returned
    ----------------
    date, ticker,
    ret_5d, ret_21d, ret_63d, ret_126d, ret_252d,
    vol_21d, vol_63d,
    ma_ratio_21d, ma_ratio_63d, ma_ratio_126d,
    target_fwd_21d          ← forward 21-day return (NaN on last rows)

    All features are computed using only past data (no lookahead).
    """
    log_ret = _log_returns(prices)
    records = []

    for ticker in prices.columns:
        px   = prices[ticker].dropna()
        lr   = log_ret[ticker].reindex(px.index)

        if len(px) < MIN_HISTORY:
            continue

        feat = pd.DataFrame(index=px.index)

        # Momentum features (simple cumulative log return)
        for w in WINDOWS:
            feat[f'ret_{w}d'] = lr.rolling(w).sum()

        # Rolling realised volatility (annualised)
        for w in VOL_WINDOWS:
            feat[f'vol_{w}d'] = lr.rolling(w).std() * np.sqrt(252)

        # Price relative to moving average
        for w in MA_WINDOWS:
            feat[f'ma_ratio_{w}d'] = px / px.rolling(w).mean() - 1

        # Forward target — shift(-21) is safe here because we only use this
        # column as a *label* after aligning on rebalance dates that lie
        # strictly in the past relative to the prediction horizon.
        feat['target_fwd_21d'] = lr.rolling(21).sum().shift(-21)

        feat['ticker'] = ticker
        feat.index.name = 'date'
        records.append(feat.reset_index())

    features_df = pd.concat(records, ignore_index=True)
    features_df = features_df.dropna(subset=[c for c in features_df.columns
                                              if c.startswith('ret_')
                                              or c.startswith('vol_')
                                              or c.startswith('ma_')])
    return features_df