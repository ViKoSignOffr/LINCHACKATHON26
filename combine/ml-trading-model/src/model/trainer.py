import pandas as pd
import lightgbm as lgb

FEATURE_COLS = [
    'ret_5d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d',
    'ma_ratio_21d', 'ma_ratio_63d', 'ma_ratio_126d',
]

LGB_PARAMS = dict(
    objective        = 'regression',
    metric           = 'rmse',
    n_estimators     = 200,
    learning_rate    = 0.05,
    num_leaves       = 31,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    verbose          = -1,
    n_jobs           = -1,
)


def get_rebalance_dates(features_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in features_df."""
    dates = pd.to_datetime(features_df['date'].unique())
    return (
        pd.Series(dates)
        .groupby(pd.to_datetime(dates).to_period('M'))
        .max()
        .sort_values()
        .values
    )


def train_model(
    features_df: pd.DataFrame,
    min_train_months: int = 24,
    prediction_horizon_days: int = 21,
) -> list[dict]:
    """
    Walk-forward expanding-window training.

    Returns a list of dicts, one per rebalance date:
        {
          'rebalance_date': pd.Timestamp,
          'model':          lgb.LGBMRegressor,
          'train_end':      pd.Timestamp,   # last date used for training
        }
    """
    features_df = features_df.copy()
    features_df['date'] = pd.to_datetime(features_df['date'])

    rebalance_dates = get_rebalance_dates(features_df)
    trading_dates = pd.DatetimeIndex(sorted(features_df['date'].unique()))
    trained_models  = []

    for i, rb_date in enumerate(rebalance_dates):
        # We need at least min_train_months of data before this rebalance
        if i < min_train_months:
            continue

        # Training set: all rows whose *target* is fully realised before rb_date.
        # target_fwd_21d at row t represents the return from t to t+21, so
        # we need t+21 <= rb_date  →  t <= rb_date - 21 observed trading days.
        rb_pos = trading_dates.searchsorted(rb_date)
        cutoff_pos = rb_pos - prediction_horizon_days
        if cutoff_pos < 0:
            continue
        cutoff = pd.Timestamp(trading_dates[cutoff_pos])
        train  = features_df[
            (features_df['date'] <= cutoff)
            & features_df['target_fwd_21d'].notna()
        ]

        if len(train) < 200:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train['target_fwd_21d']

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)

        trained_models.append({
            'rebalance_date': pd.Timestamp(rb_date),
            'model':          model,
            'train_end':      cutoff,
        })

    print(f"[trainer] Trained {len(trained_models)} walk-forward models.")
    return trained_models
