# Reproduce the ETF volatility experiment

Extract the complete ZIP into a fresh folder. Open PowerShell in the folder containing this README. The original run used 64-bit Python 3.14.7 on Windows and CPU PyTorch. With that Python available as python, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py
```

Installing dependencies requires internet access unless already available locally. Training reads the included price snapshots. Keep the folder structure, including runs/main_20261118/holdout/data/raw. Use a fresh extraction for an independent run after changing code or settings, because completed fits can be reused by the training scripts.

## Data

data/raw contains Yahoo Finance daily prices, adjusted closes, volume and corporate actions for SPY, QQQ and IWM from January 2004 through December 2025. data/manifest.json records source URLs and checksums. The three CSVs under runs/main_20261118/holdout/data/raw contain the joined history through November 6, 2026. Their corresponding manifest records the checksums and number of added sessions. get_confirmation_data.py documents the download and price adjustment at the join.

## Execution and results

reproduce.py is a new convenience launcher for the original scripts. It rebuilds the pilot in its own folder, trains and selects the main direct and residual models, evaluates the 2024–2025 exploratory and 2026 confirmation periods, verifies the forecasts, and writes the LaTeX report and interval figure under report/. The pilot's optional parallel prefit helpers are included for reference. The launcher runs the same pilot fits sequentially. Timings vary by machine.

reference_results/ contains the original selection record, rankings, per-ETF and seed scores, forecast ledger, bootstrap intervals and pilot summary tables. Use them to check your rerun. Compare numerical scores rather than timing or timestamps. The LaTeX writer includes narrative specific to the original results. Review that prose if you change data or settings.
