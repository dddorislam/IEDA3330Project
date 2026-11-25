# Reproduce Forecasting ETF Volatility and Returns

See FILE_GUIDE.md for the purpose of every script, dataset, result file, and figure.

Extract this entire archive into a fresh folder and open PowerShell there. The original run used Windows, Python 3.14.7 and CPU PyTorch. Data and forecasts are included. A new Yahoo download could revise past adjusted prices.

## Set up the environment

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt

## Check the tables using the archived forecasts

Run these four commands in order. They require no model fitting or data download.

    .\.venv\Scripts\python.exe check_report_results.py
    .\.venv\Scripts\python.exe compare_returns.py --policies-only
    .\.venv\Scripts\python.exe summarize_returns.py
    .\.venv\Scripts\python.exe link_risk_return.py

The first command checks observed variances against prices, prints the paired weekly and monthly QLIKE differences and intervals, and exports the exact training and scoring cutoffs. Table 2 uses the rankings under runs/main_20261118. Table 3 uses the ensemble forecast ledger. For Table 4 the saved per-fit timing metadata are under reference_results/timing/explore. Table 5 uses runs/return_comparison/forecast_summary.csv. Tables 6 and 7 use report/risk_return/risk_bands.csv and associations.csv. Tables 8 and 9 use strategy_summary.csv, strategy_intervals.csv and fixed_exposure_intervals.csv under runs/return_comparison. That folder also includes the pre-2026 weights, dated portfolio ledger, six signed-return forecast files and their training durations. Portfolio CSV returns and weights are in decimals. Multiply by 100 for the percentages and percentage-point differences in the report.

## Refit the models

In a separate fresh extraction, run reproduce.py with the same Python interpreter to refit the pilot and main volatility models. Then run compare_returns.py without --policies-only to refit the return models, followed by summarize_returns.py and link_risk_return.py. Full refits can take substantially longer than a table replay. The historical selection uses measured inference time to break ties, which can vary by computer. README_volatility.md gives the original volatility-run details.

## What was fixed

The workflow under docs and runs/main_20261118/lock.json document the original volatility comparison. The predeclared 0.005 tolerance is in mean QLIKE units. The return analysis, retrospective risk groups, and fixed-exposure benchmark are exploratory. The historical and extension price manifests contain SHA-256 checksums. The extension manifest records the 214 additional sessions per fund and omits the extraction timestamp. The integrated PDF, figures and LaTeX source are under report/integrated. Rebuild that source with Tectonic or a standard LaTeX installation with its listed packages.
