"""Summarize archived return forecasts and paired daily portfolio outcomes."""
from pathlib import Path

import numpy as np
import pandas as pd

def interval(values,block):
    v=np.asarray(values)
    rng=np.random.default_rng(20261122)
    ix=(rng.integers(0,len(v)-block+1,(4000,int(np.ceil(len(v)/block)),1))+np.arange(block)).reshape(4000,-1)[:,:len(v)]
    return np.quantile(v[ix].mean(axis=1),[.025,.975])

def summarize(out):
    d=pd.concat([pd.read_csv(out/f'{t}_{h}.csv') for h in [5,22] for t in ['SPY','QQQ','IWM']])
    info=[]
    for h,g in d.groupby('h'):
        base=(g['mean']-g.actual)**2
        for m in ['mean','zero','ridge','compact','reference']:
            e=g[m]-g.actual
            lossdiff=(e**2-base).groupby(g.date).mean()
            lo,hi=interval(lossdiff,20 if h==5 else 66)
            info.append(dict(h=h,model=m,n=g.date.nunique(),forecast=g[m].mean(),actual=g.actual.mean(),rmse=np.sqrt((e**2).mean()),mae=e.abs().mean(),r2=1-(e**2).mean()/base.mean(),mse_improvement_supported=hi<0,mse_diff_low=lo,mse_diff_high=hi))
    f=pd.DataFrame(info)
    f.to_csv(out/'forecast_summary.csv',index=False)
    daily = pd.read_csv(out/'daily_strategies.csv')
    for baseline, filename, column in [
        ('HAR', 'strategy_intervals.csv', 'annual_mean_diff_vs_HAR'),
        ('fixed exposure', 'fixed_exposure_intervals.csv', 'annual_mean_diff_vs_fixed'),
    ]:
        rows = []
        for horizon, group in daily.groupby('h'):
            net = group.pivot(index='exit', columns='model', values='net').sort_index()
            assert net.notna().all().all()
            for model in net:
                difference = 252 * (net[model] - net[baseline])
                low, high = interval(difference, 20)
                rows.append(dict(h=horizon, model=model, **{column: difference.mean()}, low=low, high=high))
        pd.DataFrame(rows).to_csv(out/filename, index=False)
    return f


if __name__ == '__main__':
    summarize(Path(__file__).resolve().parent/'runs'/'return_comparison')
