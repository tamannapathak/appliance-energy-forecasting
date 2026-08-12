"""
Small tests for the pipeline in scripts/run_demo_pipeline.py.

These are not exhaustive, they just check the things that would be easy
to get wrong silently: forecast lengths, MASE on a perfect forecast,
that lag features do not leak future target values, and that the
processed data has no missing target values. Run with:

    pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_demo_pipeline as pipeline


def _toy_series(n=300, freq="h"):
    index = pd.date_range("2016-01-01", periods=n, freq=freq)
    rng = np.random.default_rng(0)
    values = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 5, n)
    return pd.Series(values, index=index, name="Appliances")


def test_seasonal_naive_forecast_length_matches_horizon():
    y = _toy_series()
    horizon = 24
    index = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")

    forecast = pipeline.seasonal_naive_forecast(
        y_train=y, horizon=horizon, index=index, seasonality=pipeline.DAILY_PERIOD
    )

    assert len(forecast) == horizon
    assert list(forecast.index) == list(index)


def test_mase_is_zero_for_a_perfect_forecast():
    y_train = _toy_series()
    y_true = _toy_series(n=48).iloc[-24:]
    y_pred = y_true.copy()

    score = pipeline.mase(y_true, y_pred, y_train, seasonality=pipeline.DAILY_PERIOD)

    assert score == 0


def test_lag_features_do_not_use_future_target_values():
    df = pd.DataFrame({"Appliances": _toy_series(n=250)})

    ml_table = pipeline.make_ml_table(df, target="Appliances")

    # For every remaining row, lag_24 must equal the target value 24 rows
    # earlier in the *original* (pre-dropna) series, never a later one.
    full_target = df["Appliances"]

    for ts in ml_table.index:
        expected_lag_24 = full_target.shift(24).loc[ts]
        assert ml_table.loc[ts, "lag_24"] == expected_lag_24


def test_ml_table_has_no_lags_under_24_hours():
    # make_ml_table() is used for the 24-hour rolling backtest and must
    # only contain lag/rolling features of 24 hours or more (see its
    # docstring for why anything shorter is unsafe for a forecast that is
    # made once and held for the next 24 hours).
    df = pd.DataFrame({"Appliances": _toy_series(n=400)})

    ml_table = pipeline.make_ml_table(df, target="Appliances")

    lag_cols = [c for c in ml_table.columns if c.startswith("lag_")]
    roll_cols = [c for c in ml_table.columns if c.startswith("roll_")]

    assert lag_cols == ["lag_24", "lag_48", "lag_168"]
    for col in roll_cols:
        window = int(col.rsplit("_", 1)[-1])
        assert window >= 24


def test_rolling_features_do_not_leak_within_a_forecast_block():
    # Bug fix: an earlier version of make_ml_table() computed rolling
    # features as y.shift(1).rolling(window), which still ends at t-1
    # for every row. That's safe for the first hour of a 24-hour forecast
    # block, but for a later hour in the same block it reaches into
    # values earlier in that block, which are not actually known yet
    # when the whole 24-hour forecast is made once at the block's start.
    # make_ml_table() now computes rolling features once per block (using
    # only data strictly before the block starts) and broadcasts them to
    # every row in the block. This test checks that directly: changing
    # the realised values *inside* a forecast block must not change that
    # block's rolling features at all.
    horizon = 24
    test_steps = 48
    df = pd.DataFrame({"Appliances": _toy_series(n=500)})

    ml_table = pipeline.make_ml_table(df, target="Appliances", horizon=horizon, test_steps=test_steps)
    first_test_block = ml_table.iloc[-test_steps:].iloc[:horizon]

    roll_cols = [c for c in ml_table.columns if c.startswith("roll_")]
    for col in roll_cols:
        assert first_test_block[col].nunique() == 1, f"{col} is not constant within a single forecast block"

    corrupted = df.copy()
    within_block_index = first_test_block.index[1:]
    corrupted.loc[within_block_index, "Appliances"] = 1e9

    corrupted_table = pipeline.make_ml_table(corrupted, target="Appliances", horizon=horizon, test_steps=test_steps)
    corrupted_block = corrupted_table.iloc[-test_steps:].iloc[:horizon]

    for col in roll_cols:
        assert np.allclose(first_test_block[col].values, corrupted_block[col].values), (
            f"{col} changed after corrupting values inside its own forecast block - it is leaking future information"
        )


def test_tune_feature_model_returns_a_param_dict_fit_feature_model_accepts():
    df = pd.DataFrame({"Appliances": _toy_series(n=500)})
    ml_table = pipeline.make_ml_table(df, target="Appliances")
    feature_cols = [c for c in ml_table.columns if c != "Appliances"]

    best_params, results = pipeline.tune_feature_model(
        X_train=ml_table[feature_cols],
        y_train=ml_table["Appliances"],
        n_val_steps=48,
    )

    assert set(best_params) == {"n_estimators", "learning_rate", "max_depth"}
    assert len(results) > 0

    # Should not raise: fit_feature_model must accept exactly this dict.
    pipeline.fit_feature_model(ml_table[feature_cols], ml_table["Appliances"], params=best_params)


def test_processed_hourly_data_has_no_missing_target_values(tmp_path):
    # Build a tiny fake raw CSV in the shape load_appliance_data() expects,
    # rather than depending on the real (large) dataset being present.
    dates = pd.date_range("2016-01-01", periods=600, freq="10min")
    fake = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "Appliances": np.random.default_rng(1).uniform(30, 200, len(dates)),
        "T_out": np.random.default_rng(2).uniform(0, 15, len(dates)),
    })

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    fake.to_csv(raw_dir / "energydata_complete.csv", index=False)

    original_data_dir = pipeline.DATA_DIR
    try:
        pipeline.DATA_DIR = tmp_path / "data"
        hourly = pipeline.load_appliance_data()
    finally:
        pipeline.DATA_DIR = original_data_dir

    assert hourly["Appliances"].isna().sum() == 0
