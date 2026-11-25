# ETF volatility forecasting workflow

## 1. Start with a fresh pilot run

Run the existing pipeline from data audit through evaluation in a new folder. The pilot checks that the pipeline works end to end and helps identify candidates for the main experiment. Pilot scores are preliminary and support no final accuracy claim.

Use SPY, QQQ and IWM, daily data from 2004 through 2025, and the existing 5- and 22-trading-day horizons. The 2004 data provide feature history before training begins in 2005. Use the archived raw snapshot by default so differences between pilot runs can be traced to code and settings. To test data retrieval too, omit the raw snapshot copy and let the download script retrieve the same date range. Record which option was used and retain the source hashes and retrieval metadata.

From the course workspace in PowerShell, run the following with the project's Python environment already installed. See [the project README](etf_volatility/README.md#reproduce) for environment setup. The timestamped folder starts without saved model weights, processed data or scores.

```powershell
$project = Resolve-Path 'etf_volatility'
$python = Join-Path $project.Path '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Set up the project Python environment first.' }

$pilot = Join-Path $project.Path ('runs/pilot-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $pilot -ErrorAction Stop | Out-Null
$pilotFiles = @('config.json', 'download_data.py', 'run_experiment.py', 'evaluate.py', 'verify_results.py', 'requirements-lock.txt')
foreach ($file in $pilotFiles) {
    Copy-Item -LiteralPath (Join-Path $project.Path $file) -Destination $pilot -ErrorAction Stop
}

# Use the archived data snapshot. Omit this block to test a fresh download.
$pilotData = Join-Path $pilot 'data'
New-Item -ItemType Directory -Path $pilotData -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath (Join-Path $project.Path 'data/raw') -Destination $pilotData -Recurse -ErrorAction Stop
Copy-Item -LiteralPath (Join-Path $project.Path 'data/manifest.json') -Destination (Join-Path $pilotData 'source_manifest.json') -ErrorAction Stop

function Invoke-PilotStep {
    param([string]$Script, [string[]]$Arguments = @())
    & $python (Join-Path $pilot $Script) @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pilot step failed: $Script ($LASTEXITCODE)" }
}

Invoke-PilotStep -Script 'download_data.py'
Invoke-PilotStep -Script 'run_experiment.py' -Arguments @('--stage', 'check')
foreach ($h in @(5, 22)) {
    Invoke-PilotStep -Script 'run_experiment.py' -Arguments @('--stage', 'validate', '--horizon', "$h")
    Invoke-PilotStep -Script 'run_experiment.py' -Arguments @('--stage', 'test', '--horizon', "$h")
}
Invoke-PilotStep -Script 'evaluate.py'
Invoke-PilotStep -Script 'verify_results.py'
Write-Output "Pilot artifacts saved in $pilot"
```

Before moving on, confirm complete trading-session coverage, causal features, correct target boundaries, finite positive forecasts, model reloads and reproducible scores. Inspect learning curves and compare sample forecasts with HAR on matching dates. Record failures, fixes, runtime and candidate ideas in a short pilot note. Treat the runner's `test` output for 2024–2025 as a pilot evaluation. Keep that period available for exploratory comparisons.

## 2. Set up the main experiment

Forecast after the close of session $t$ using information available by that close. Fit a separate model per ETF and forecast horizon $h\in\{5,22\}$. Use adjusted closing prices $P_t$ for log returns and define the target as the average squared return over the next $h$ sessions.

$$
r_t=\log(P_t/P_{t-1}),\qquad v_{t,h}=\frac{1}{h}\sum_{j=1}^{h}r_{t+j}^2,\qquad \sigma_{t,h}=\sqrt{252v_{t,h}}.
$$

Keep seven features initially: return, absolute return, logs of trailing 5-, 22- and 60-session average squared returns (each with $10^{-8}$ added before taking the log), log high/low, and $\log(1+V_t)$ minus its trailing 22-session mean. Each window ends at the forecast origin. Fit feature scaling and any target scaling on eligible training data within each fold.

Use expanding chronological folds with validation in 2017–2018, 2019–2020 and 2021–2023. Train on earlier data starting in 2005, removing any training target that ends on or after the first validation session. Score only validation origins whose complete target ends within the validation period. Use the same eligible forecast origins across candidates being compared. For direct versus hybrid comparisons, also fit a direct reference on the hybrid's eligible training origins.

## 3. Compare simpler direct LSTMs

Evaluate each change separately. Keep the feature set, training budget and scoring rules fixed for the architecture comparisons.

| Candidate | Change being tested |
|---|---|
| Reference direct LSTM | 60-session window, 32 hidden units, 16-unit intermediate dense layer |
| Shorter window | Use 22 sessions with the same layers |
| Simpler output head | Keep 22 sessions and 32 hidden units. Map the final LSTM state directly to one output |
| Narrower LSTM | Keep 22 sessions and the simpler output head. Reduce the hidden state to 16 units |

Use dropout 0.1 consistently before the output head for this controlled comparison. An additional reproduction of the pilot-selected architecture can be reported as context. For direct forecasts, use a Softplus output plus a small positive constant, then undo the target scaling to obtain $\widehat v_{t,h}$.

Train with Adam, learning rate 0.001, batch size 32, a limit of 100 epochs, validation patience of 10 epochs and gradient norm clipping at 1.0. In each batch, the LSTM processes the window in order, forecasts the variance, computes QLIKE and updates its weights through backpropagation through time. Save the checkpoint with the lowest validation QLIKE. Use seeds 17, 42 and 123 for each neural candidate and average their predictions on the variance scale before converting to volatility. Use the median best epoch across folds for each final ETF and seed fit.

## 4. Test HAR with an LSTM correction

Fit a HAR-style baseline using daily, weekly and monthly squared-return averages $q_{t,k}=k^{-1}\sum_{j=0}^{k-1}r_{t-j}^2$.

$$
b_{t,h}=\max(10^{-8},\ \beta_0+\beta_1q_{t,1}+\beta_5q_{t,5}+\beta_{22}q_{t,22}).
$$

Use 2005–2009 for the initial HAR history. Generate a dated record of issued forecasts from 2010 onward. Refit HAR at each month end with outcomes that have fully matured by that close and apply the new coefficients from the next trading session. Use this same schedule when building historical correction inputs, validating, and evaluating. Keep the correction weights and scaler fixed within each evaluation block. Use the seven features plus the log of the issued HAR prediction for each day in the input window. Start with a 22-session window, 16 hidden units and a scalar output initialized to zero.

$$
g_t=\log(4)\tanh\!\left(\frac{z_t}{\log(4)}\right),\qquad
\widehat v_{t,h}=\max(10^{-8},\ b_{t,h}e^{g_t}).
$$

The initial correction is zero, so the first combined forecast equals HAR. Train with full QLIKE plus $\lambda g_t^2$, testing $\lambda\in\{0,0.01,0.1\}$ on validation. Compare the hybrid with the same monthly HAR forecast, HAR with a constant multiplier, HAR with a regularized linear correction, and the direct LSTMs. Fit the constant and linear corrections using eligible historical forecasts only. Check that truncating the input history at any origin reproduces its features and issued baseline. The [detailed residual specification](ETF_LSTM_workflow_residual_revised.md#71-revised-extension--har-style-residual-learning) describes the optional delayed residual feature and additional numerical checks.

## 5. Select the models and confirm performance

Score forecasts on the original variance scale with $\operatorname{QLIKE}(v,\widehat v)=\log(\widehat v)+v/\widehat v$. Lower is better. Report annualized volatility MAE and RMSE in percentage points, parameter count, training time and inference time on the same hardware. Compare models separately by horizon using paired dates, equal weighting across ETFs and folds for selection, and individual ETF and seed results for context.

Select the least costly direct candidate within 0.005 mean validation QLIKE of the best direct model. Declare the cost rule before fitting: prefer fewer parameters, then lower measured inference time. Use validation to select the hybrid penalty. Use 2024–2025 for exploratory checks and record any design changes they motivate. Before final scoring, lock the selected models, primary comparisons (simplified direct versus reference direct, and hybrid versus scheduled HAR), test dates, metrics and uncertainty method. Set this holdout aside before the main experiment. An unexamined portion of 2026 is a candidate if complete outcomes are available and no results from that period have guided development. Otherwise reserve a later period. Include only origins with fully matured targets inside the reserved block.

On the reserved period, report candidate-minus-reference mean QLIKE and paired block-bootstrap intervals. Resample common date blocks across ETFs to account for overlapping targets and shared market moves. An interval entirely below zero supports improved accuracy. An upper bound below the predeclared 0.005 tolerance supports accuracy preservation. Compare runtime and size separately so the report can show whether a simpler model also reduces cost.

## 6. Save the experiment record

- Pilot data audit, verification JSON, prediction tables, scores and a short pilot note.
- Main experiment configuration, validation results, training curves, saved models and scalers, plus the selected models and holdout lock record.
- Forecast ledger with ETF, horizon, origin, target end, model, seed, observed variance and predicted variance. For HAR-based models, also retain the issued baseline, refit cutoff and correction multiplier.
- Final comparison table, paired intervals, observed-versus-predicted plots and measured costs. Keep the pilot, exploratory comparisons and final results in separately labelled run folders.

The commands in Section 1 use the existing pilot runner. Sections 2–6 specify the main experiment. The scalar output head, scheduled HAR corrections and fresh confirmation require implementation in the main experiment runner before execution.
