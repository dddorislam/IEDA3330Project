# Executed ETF volatility experiment

This project implements `../ETF_LSTM_workflow.md` for SPY, QQQ and IWM. Its primary target is five-trading-day forward average squared log returns, with a separately tuned 22-day robustness experiment. It uses historical data through 2025, not live predictions.

## Outputs

- `report/ETF_volatility_results.pdf`: empirical report with methods, comparisons and figures.
- `report/results_summary.md`: short results summary.
- `outputs/h5/` and `outputs/h22/`: selection records, validation curves, all test predictions, metrics, non-overlapping evaluation, yearly results, calibration, block-bootstrap intervals, model weights and transformations.
- `data/raw/`: downloaded price snapshots and action audits; preserved rather than overwritten on rerun.
- `data/manifest.json`: retrieval timestamp, source URLs, hashes and download-library versions.
- `data/audit.csv`: trading-session coverage and corporate-action counts.
- `outputs/verification.json`: label, split and causal-feature checks.
- `environment.json` and `requirements-lock.txt`: execution environment versions.

## Reproduce

Use Python 3.14 or an environment compatible with the locked packages. The initial local venv could not bootstrap pip, so pip from the host interpreter was used to install into it. The project interpreter itself works normally.

From the course workspace in PowerShell:

```powershell
python -m venv etf_volatility/.venv
python -m pip --python etf_volatility/.venv/Scripts/python.exe install -r etf_volatility/requirements-lock.txt
etf_volatility/.venv/Scripts/python.exe etf_volatility/download_data.py
etf_volatility/.venv/Scripts/python.exe etf_volatility/run_experiment.py --stage check
etf_volatility/.venv/Scripts/python.exe etf_volatility/run_experiment.py --stage validate --horizon 5
etf_volatility/.venv/Scripts/python.exe etf_volatility/run_experiment.py --stage validate --horizon 22
etf_volatility/.venv/Scripts/python.exe etf_volatility/run_experiment.py --stage test --horizon 5
etf_volatility/.venv/Scripts/python.exe etf_volatility/run_experiment.py --stage test --horizon 22
etf_volatility/.venv/Scripts/python.exe etf_volatility/evaluate.py
etf_volatility/.venv/Scripts/python.exe etf_volatility/verify_results.py
etf_volatility/.venv/Scripts/python.exe etf_volatility/build_report.py
etf_volatility/.venv/Scripts/python.exe etf_volatility/predict.py --date 2025-12-31
```

Data downloading needs network access. Cached raw files are read without downloading again. To preserve the original result, run changed configurations/code in a fresh copy with a new output directory: the fitting stages resume cached completed runs and are not a cache-invalidation system. Do not mix outputs produced by different configurations or source revisions. Final test training checks the frozen configuration hash against the selection record.

`predict.py` reloads saved core models and averages their variance forecasts. Its default origin is the last cached historical date, 2025-12-31; it does not fetch live data or produce a current-market forecast. `outputs/historical_origin_forecasts.json` records a demonstration run at that origin.

## Protocol

The configuration was fixed before final-test metrics were calculated. Validation folds are 2017–2018, 2019–2020 and 2021–2023 with expanding prior training data. All target labels crossing a split boundary are removed. The 2024–2025 final test uses frozen model parameters; daily features update with observed data. Warm-up and common-date eligibility use the maximum candidate lookback so candidate scores are comparable.

Eight configurations per horizon are compared using seed 17. The selected common architecture is evaluated on all validation folds with seeds 17, 42 and 123. Each final ETF/seed fit uses its median best validation epoch. Three predictions are averaged on the variance scale. A predeclared ablation excludes range and volume, keeping the core-selected architecture but selecting its own epoch counts. VIX and pooled-asset extensions are not part of this run.

QLIKE is `log(predicted_variance) + actual_variance / predicted_variance`; it can be negative and lower is better. MAE/RMSE are annualized volatility errors in percentage points. Bootstrap loss differences are LSTM ensemble minus comparator, so negative favors LSTM. Intervals are conditional on the fitted models and do not adjust for all multiple comparisons or training uncertainty. Macro scores give equal weight to each ETF, not each asset's market capitalization.

The GARCH implementation fits zero-mean Gaussian GARCH(1,1) on percent returns, recursively filters new observations using fixed parameters, averages the h future conditional variances and divides by 10,000. The regression is HAR-style using daily proxies, not original intraday HAR-RV. EWMA uses a fixed decay of 0.94 and the 2004 warm-up history. No price observations are forward-filled.

## Data provenance and spot checks

Yahoo history has split-adjusted historical `Close` values; `Adj Close` additionally accounts for distributions. Returns use `Adj Close`, and high/low ratios use the consistent same-day OHLC fields. Corporate actions are retained for audit rather than applied again.

- The SPY 2024-12-20 distribution is 1.966 USD in the export, matching issuer disclosure 1.965548 USD to export precision: https://www.ssga.com/library-content/products/fund-docs/etfs/apac/jp/Securities_Report_SPY.pdf
- The IWM 2-for-1 split is dated 2005-06-09 in the export and an SEC-filed instrument description: https://www.sec.gov/Archives/edgar/data/19617/000089109210005752/e41298_424b2.pdf

These limited checks do not replace comprehensive multi-vendor verification. Historical data may be revised; raw hashes anchor this run. The data describe a small selected surviving US equity ETF universe and should not be generalized to all ETFs. No trading or investment performance was tested.
