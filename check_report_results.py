"""Recompute the headline volatility comparisons and evaluation dates from the snapshots."""
from pathlib import Path

import numpy as np
import pandas as pd

from write_main_report import diff_ci


def main():
    root = Path(__file__).resolve().parent
    run = root / 'runs' / 'main_20261118'
    ledger = pd.read_csv(run / 'forecast_ledger.csv')
    scores = ledger.loc[ledger.stage == 'confirm'].copy()
    scores['qlike'] = np.log(scores.predicted_variance) + scores.actual_variance / scores.predicted_variance
    np.testing.assert_allclose(scores.qlike, scores.loss)
    for horizon, model, reference, block in [(5, 'short', 'reference', 20),
                                             (22, 'hybrid1', 'HAR', 66),
                                             (22, 'hybrid1', 'linear', 66)]:
        paired = scores[scores.horizon == horizon].pivot(
            index=['origin_date', 'ticker'], columns='ensemble', values='qlike')
        difference, low, high = diff_ci(paired[model], paired[reference], block)
        print(f'{horizon} days, {model} minus {reference}: {difference:.4f} [{low:.4f}, {high:.4f}]')

    blocks = [('Validation 1', '2017-01-01', '2018-12-31'),
              ('Validation 2', '2019-01-01', '2020-12-31'),
              ('Validation 3', '2021-01-01', '2023-12-31'),
              ('Exploratory', '2024-01-01', '2025-12-31'),
              ('Confirmation', '2026-01-01', '2026-11-06')]
    cutoffs = []
    for fund in ['SPY', 'QQQ', 'IWM']:
        prices = pd.read_csv(run / 'holdout' / 'data' / 'raw' / f'{fund}.csv', index_col='date')
        dates = prices.index
        squared_returns = np.log(prices['Adj Close']).diff().pow(2)
        for horizon in [5, 22]:
            observed = squared_returns.rolling(horizon).mean().shift(-horizon)
            recorded = scores[(scores.ticker == fund) & (scores.horizon == horizon) & (scores.ensemble == 'HAR')]
            np.testing.assert_allclose(observed.loc[recorded.origin_date], recorded.actual_variance)
            for stage, start, end in blocks:
                first = dates.searchsorted(start)
                last = dates.searchsorted(end, side='right') - 1
                cutoffs.append([fund, stage, horizon, dates[first-1-horizon], dates[first-1],
                                dates[first], dates[last-horizon], dates[last]])
    columns = ['ETF', 'stage', 'horizon', 'last_training_origin', 'last_training_outcome',
               'first_scored_origin', 'last_scored_origin', 'last_scored_outcome']
    pd.DataFrame(cutoffs, columns=columns).to_csv(run / 'evaluation_cutoffs.csv', index=False)


if __name__ == '__main__':
    main()
