# Appliance Energy Forecasting

Coursework project forecasting household appliance energy use using the
UCI **Appliances Energy Prediction** dataset. Benchmark models, SARIMAX,
an XGBoost feature-based model, and a time-series foundation model are
fitted and compared on a held-out 14-day test period, forecast 24 hours
at a time.

## Project aim

Forecast short-term household appliance energy use and check whether
increasingly complex models actually improve on simple benchmarks.

1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model improve performance?
5. Does a time-series foundation model add anything on top of that?
6. Which model would actually be sensible for a smart-home system?

## Dataset

`data/raw/energydata_complete.csv` — the Appliances Energy Prediction
dataset (Candanedo et al., 2017), 19,735 rows sampled every 10 minutes
from 2016-01-11 to 2016-05-27, no missing values. Columns include the
target (`Appliances`, Wh), `lights`, nine indoor temperature/humidity
pairs (`T1`..`T9`, `RH_1`..`RH_9`), outdoor weather (`T_out`,
`Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`), and two
random variables (`rv1`, `rv2`) included by the original dataset authors
as a noise check, not used here.

The pipeline resamples this to hourly means (3,290 hourly rows) before
modelling — see the report for why.

## Forecasting task

- **Target:** `Appliances`, hourly mean.
- **Test period:** final 14 days (336 hours); everything before that is
  training data.
- **Horizon:** 24 hours, evaluated as 14 separate daily origins across
  the 336-hour test period (`rolling_backtest()` in
  `scripts/run_demo_pipeline.py`) — this is the horizon the brief and
  this README's "Forecasting task" section specify, and every model is
  evaluated this way.
- **Metrics:** MAE, RMSE, MASE (scaled against the in-sample daily
  seasonal naive error), Bias.

## Models

1. **Benchmarks** — mean, naive, daily seasonal naive (lag 24), weekly
   seasonal naive (lag 168), drift. Recomputed at each of the 14 daily
   origins from the real, expanding history.
2. **SARIMAX** — order chosen with an AIC grid search over
   p=[0,6], d=[0,2], q=[0,6] (147 fits), seasonal order fixed at
   (1,1,1,24), exogenous variables `T_out`, `RH_out`, `Windspeed`,
   `Visibility`, `Tdewpoint`. Fitted once, then rolled forward daily with
   `.append(refit=False)`.
3. **Feature-based model** — XGBoost (`XGBRegressor`) on lag, rolling,
   time-of-day, day-of-week, indoor sensor and outdoor weather features.
   Only lag/rolling features of 24 hours or more are used
   (`make_ml_table()`), since anything shorter is not genuinely known a
   full 24 hours ahead of a daily forecast origin. Hyperparameters
   (`n_estimators`, `learning_rate`, `max_depth`) are chosen by
   `tune_feature_model()`, a small validation-split search, rather than
   fixed by hand.
4. **Foundation model** — Chronos (`amazon/chronos-t5-tiny`) is tried
   first, zero-shot; if it can't run, TimeGPT (Nixtla's hosted API) is
   tried next. Neither could actually run in the development environment
   (see Limitations below); the pipeline falls back to the daily
   seasonal naive forecast and logs which of the three actually happened.

## Results

The results table is `outputs/metrics/model_comparison.csv`, written
fresh by `python scripts/run_demo_pipeline.py` — this README does not
hardcode numbers here so that it never goes stale against a real run.
The full discussion is in `reports/report.docx`.

## Repository structure

```text
appliance-energy-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/energydata_complete.csv
│   └── processed/appliance_hourly.csv      (written by the pipeline)
│
├── scripts/
│   └── run_demo_pipeline.py                (the whole pipeline)
│
├── notebooks/
│   └── run_demo_pipeline.ipynb             (same pipeline, cell-by-cell)
│
├── outputs/
│   ├── figures/
│   ├── forecasts/all_forecasts.csv
│   └── metrics/
│
├── reports/
│   └── report.docx
│
└── tests/
    └── test_pipeline.py
```

Everything is kept in the single `scripts/run_demo_pipeline.py` file
rather than split into a `src/` package, following the structure of the
tutor-provided starter script that this was built on top of, and because
the assignment brief asked for the answers to be added directly into
that file.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the pipeline

```bash
python scripts/run_demo_pipeline.py
```

This loads `data/raw/energydata_complete.csv` (falls back to downloading
it from UCI if that file is missing), resamples to hourly data, runs the
EDA and stationarity checks, fits all four model classes, evaluates them
with a 24-hour rolling backtest, and writes everything below.

A full run takes a few minutes on a normal laptop — most of the time
goes into the SARIMAX order grid search (147 fits) and the final
SARIMAX fit with exogenous variables. `scripts/run_demo_pipeline.py`
checkpoints the grid search to
`outputs/metrics/sarimax_order_search.csv` and skips any (p, d, q)
combination already recorded there, so a second run does not repeat
finished work.

## Outputs

- `outputs/forecasts/all_forecasts.csv` — actual values and every
  model's forecast for the test period.
- `outputs/metrics/model_comparison.csv` — MAE, RMSE, MASE, Bias per
  model.
- `outputs/metrics/model_comparison_summary.txt` — plain-English
  comparison against the strongest benchmark.
- `outputs/metrics/sarimax_coefficients.csv` — fitted SARIMAX
  coefficients, standard errors and p-values.
- `outputs/metrics/feature_model_tuning.csv` — the hyperparameter search
  results from `tune_feature_model()`.
- `outputs/metrics/feature_importance.csv` — permutation importance for
  the feature-based model.
- `outputs/metrics/foundation_model_notes.txt` — records which of
  Chronos / TimeGPT / the fallback actually ran.
- `outputs/metrics/sarimax_order_search.csv` — every (p, d, q, AIC)
  combination tried in the grid search.
- `outputs/metrics/stationarity_tests.csv` — ADF/KPSS results.
- `outputs/figures/` — EDA plots, seasonal decomposition, ACF/PACF,
  SARIMAX residual diagnostics and confidence interval plot, feature
  importance, forecast comparison, error diagnostics.

## Data leakage and conditional forecasts

Two things in this pipeline would not genuinely be known at a real
forecast origin in operational use, and both are flagged with a printed
note when the pipeline runs:

- The SARIMAX exogenous variables (`T_out`, `RH_out`, `Windspeed`,
  `Visibility`, `Tdewpoint`) use the *realised* weather for each 24-hour
  block, not a weather forecast for it.
- `make_ml_table()` only uses lag/rolling features of 24 hours or more,
  so every feature the XGBoost model sees is genuinely known at each
  daily forecast origin — this was a deliberate design choice to avoid
  the leakage that shorter lags would introduce (see the function's
  docstring).

See the report's answer to Question 5 for the full discussion.

## Limitations

Neither foundation model could be run in the environment this project
was developed in. Chronos itself installs cleanly under the pinned
`torch==2.2.2` / `chronos-forecasting==1.5.3` combination in
`requirements.txt` (well under 200 MB, no CUDA packages), but it still
needs to download its model weights from `huggingface.co` on first use,
and that domain was blocked by network policy in this environment.
TimeGPT needs a Nixtla API key (`NIXTLA_API_KEY`), which was not
available either. `get_foundation_model_forecast_rolling()` in
`scripts/run_demo_pipeline.py` tries both, one 24-hour-ahead call per
daily origin (matching how every other model here is evaluated, and
keeping within Chronos's own recommended prediction length), and falls
back to the same rolling daily seasonal naive computation used for the
`seasonal_naive_daily` benchmark if both fail — logging clearly which
one actually happened, rather than the pipeline failing outright or
guessing at a number. This is discussed further in the report.

## Tests

```bash
pytest
```

Covers: forecast length matches the requested horizon, MASE is zero for
a perfect forecast, lag features never use a future target value, the
feature table never contains a lag or rolling window under 24 hours,
`tune_feature_model()`'s output is a dict `fit_feature_model()` actually
accepts, and the processed hourly data has no missing target values.
