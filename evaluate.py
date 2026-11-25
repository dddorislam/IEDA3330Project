"""Summaries and synchronized moving-block uncertainty from saved forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'config.json').read_text())
def metrics(g, reference):
    y=g.actual_variance.to_numpy(); p=g.predicted_variance.to_numpy()
    assert np.isfinite(p).all() and (p>0).all()
    ref=reference.reindex(g.origin_date).to_numpy()
    diff=100*np.sqrt(252*p)-100*np.sqrt(252*y)
    return dict(n=len(g),qlike=float(np.mean(np.log(p)+y/p)),vol_mae_pp=float(np.abs(diff).mean()),vol_rmse_pp=float(np.sqrt(np.mean(diff**2))),variance_mse=float(np.mean((y-p)**2)),oos_r2=float(1-np.sum((y-p)**2)/np.sum((y-ref)**2)))

for h in C['horizons']:
    base=ROOT/f'outputs/h{h}'
    if not (base/'test_predictions.csv').exists(): continue
    d=pd.read_csv(base/'test_predictions.csv',parse_dates=['origin_date','label_end_date'])
    assert not d.duplicated(['ticker','origin_date','model','seed']).any()
    rows=[]; nonoverlap=[]; regimes=[]
    for (ticker,model,seed),g in d.groupby(['ticker','model','seed']):
        g=g.sort_values('origin_date')
        ref=d.loc[(d.ticker==ticker)&(d.model=='Historical22')].set_index('origin_date').predicted_variance
        assert set(g.origin_date)==set(ref.index)
        rows.append(dict(ticker=ticker,model=model,seed=seed,**metrics(g,ref)))
        nonoverlap.append(dict(ticker=ticker,model=model,seed=seed,offset=0,**metrics(g.iloc[::h],ref)))
        for year in [2024,2025]:
            gy=g.loc[g.origin_date.dt.year==year]
            regimes.append(dict(ticker=ticker,model=model,seed=seed,year=year,**metrics(gy,ref)))
    m=pd.DataFrame(rows); m.to_csv(base/'metrics.csv',index=False)
    pd.DataFrame(nonoverlap).to_csv(base/'nonoverlapping_metrics.csv',index=False)
    pd.DataFrame(regimes).to_csv(base/'year_metrics.csv',index=False)
    # Average per-ETF metrics (RMSE here is macro-average, not pooled RMSE).
    m.groupby(['model','seed'])[['qlike','vol_mae_pp','vol_rmse_pp','oos_r2']].mean().sort_values('qlike').to_csv(base/'macro_metrics.csv')
    d['loss']=np.log(d.predicted_variance)+d.actual_variance/d.predicted_variance
    ensemble=d.loc[d.model=='LSTM-ensemble'].pivot(index='origin_date',columns='ticker',values='loss').sort_index()
    boot=[]; rng=np.random.default_rng(20261107)
    for model in ['Historical5','Historical22','EWMA','GARCH','HAR-style','LSTM-ablation-ensemble']:
        other=d.loc[d.model==model].pivot(index='origin_date',columns='ticker',values='loss').reindex(index=ensemble.index,columns=ensemble.columns)
        delta=ensemble-other
        assert not delta.isna().any().any()
        for length in (C['bootstrap_blocks'] if h==5 else [44,66,88]):
            n=len(delta); number=int(np.ceil(n/length))
            starts=rng.integers(0,n-length+1,size=(C['bootstrap_replicates'],number))
            ix=(starts[:,:,None]+np.arange(length)).reshape(C['bootstrap_replicates'],-1)[:,:n]
            for ticker in list(delta.columns)+['Equal-weight ETFs']:
                series=delta[ticker].to_numpy() if ticker in delta else delta.mean(axis=1).to_numpy()
                means=series[ix].mean(axis=1)
                lo,hi=np.quantile(means,[.025,.975])
                boot.append(dict(ticker=ticker,baseline=model,block_length=length,n=n,mean_loss_difference=series.mean(),ci_low=lo,ci_high=hi,replicates=C['bootstrap_replicates']))
    pd.DataFrame(boot).to_csv(base/'block_bootstrap.csv',index=False)
    # Calibration bins are descriptive test diagnostics, never tuning inputs.
    cal=[]
    for ticker in C['tickers']:
        g=d.loc[(d.ticker==ticker)&(d.model=='LSTM-ensemble')].copy()
        g['bin']=pd.qcut(g.predicted_variance,5,labels=False,duplicates='drop')
        for b,gg in g.groupby('bin'):
            cal.append(dict(ticker=ticker,bin=int(b),n=len(gg),mean_predicted_variance=gg.predicted_variance.mean(),mean_actual_variance=gg.actual_variance.mean()))
    pd.DataFrame(cal).to_csv(base/'calibration.csv',index=False)
    print('HORIZON',h,flush=True)
    print(m.loc[m.model.isin(['LSTM-ensemble','GARCH','EWMA','Historical22']),['ticker','model','qlike','vol_rmse_pp','oos_r2']].to_string(index=False),flush=True)
