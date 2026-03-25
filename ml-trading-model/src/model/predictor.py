import pandas as pd
from src.model.trainer import FEATURE_COLS

def make_predictions(
    trained_models: list[dict],
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each rebalance date, score every stock available on that date.

    Returns
    -------
    DataFrame with columns: [rebalance_date, ticker, score, target_fwd_21d]
    """
    features_df = features_df.copy()
    features_df['date'] = pd.to_datetime(features_df['date'])

    rows = []
    for entry in trained_models:
        rb_date = entry['rebalance_date']
        model   = entry['model']

        # Snapshot: features as of the rebalance date (last trading day of month)
        snap = features_df[features_df['date'] == rb_date].copy()
        if snap.empty:
            # Fall back to closest prior date within 5 calendar days
            window = features_df[
                (features_df['date'] <= rb_date)
                & (features_df['date'] >= rb_date - pd.Timedelta(days=5))
            ]
            if window.empty:
                continue
            latest = window['date'].max()
            snap   = features_df[features_df['date'] == latest].copy()

        valid = snap.dropna(subset=FEATURE_COLS)
        if valid.empty:
            continue

        scores = model.predict(valid[FEATURE_COLS])
        valid  = valid.copy()
        valid['score']          = scores
        valid['rebalance_date'] = rb_date
        rows.append(valid[['rebalance_date', 'ticker', 'score', 'target_fwd_21d']])

    if not rows:
        return pd.DataFrame(columns=['rebalance_date', 'ticker', 'score', 'target_fwd_21d'])

    return pd.concat(rows, ignore_index=True)