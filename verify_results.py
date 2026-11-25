"""Independent artifact checks, including reloading a model and GARCH recursion."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from arch import arch_model
from run_experiment import ROOT,C,Net,prepare,indices,FEATURES,qlike,baseline

checks=[]
for h in C['horizons']:
    base=ROOT/f'outputs/h{h}'
    saved=pd.read_csv(base/'test_predictions.csv',parse_dates=['origin_date','label_end_date'])
    for ticker in C['tickers']:
        d=prepare(ticker,h); start,end=C['test'];tr=indices(d,start,end,True);ev=indices(d,start,end)
        path=base/'models'/f'{ticker}_core_seed17'
        meta=json.loads(path.with_suffix('.json').read_text()); hp=meta['hp']; sc=meta['scaler']; L=hp['lookback']
        a=d[sc['features']].to_numpy(float)
        rows=np.unique(np.concatenate([np.arange(i-L+1,i+1) for i in tr]))
        assert np.allclose(a[rows].mean(axis=0),sc['mean'])
        assert np.isclose(d.target.iloc[tr].mean(),sc['variance_scale'])
        scaled=(a-np.array(sc['mean']))/np.array(sc['std'])
        model=Net(len(sc['features']),hp['hidden'],hp['dropout'])
        model.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True));model.eval()
        picked=ev[[0,100,-1]]
        x=torch.tensor(np.stack([scaled[i-L+1:i+1] for i in picked]),dtype=torch.float32)
        with torch.no_grad(): pred=model(x).numpy()*sc['variance_scale']
        reference=saved.loc[(saved.ticker==ticker)&(saved.model=='LSTM')&(saved.seed==17)].set_index('origin_date')
        assert np.allclose(pred,reference.loc[d.index[picked],'predicted_variance'],rtol=1e-5)
        # The independently reconstructed actual target matches every saved record.
        for date,row in reference.iterrows():
            i=d.index.get_loc(date)
            assert np.isclose(row.actual_variance,np.mean(d['return'].iloc[i+1:i+h+1].to_numpy()**2),rtol=1e-8)
            assert row.label_end_date==d.index[i+h]
        # Compare GARCH implementation to arch's fixed-parameter analytic forecast.
        r=100*d['return'].dropna();r=r.loc[r.index>=C['train_start']]
        trainr=r.loc[r.index<=d.index[tr[-1]]]
        fitted=arch_model(trainr,mean='Zero',vol='GARCH',p=1,q=1,rescale=False).fit(disp='off',show_warning=False)
        fixed=arch_model(r,mean='Zero',vol='GARCH',p=1,q=1,rescale=False).fix(fitted.params)
        af=fixed.forecast(horizon=h,start=d.index[ev[0]],reindex=False).variance.mean(axis=1)/10000
        reference_g=saved.loc[(saved.ticker==ticker)&(saved.model=='GARCH')].set_index('origin_date').predicted_variance
        assert np.allclose(af.reindex(reference_g.index),reference_g,rtol=1e-5,atol=1e-10)
        checks.append(dict(ticker=ticker,horizon=h,model_reload=True,scaler_training_only=True,all_targets_recomputed=True,garch_matches_arch_analytic=True,test_n=len(ev)))
    # Recalculate every reported QLIKE directly from per-origin records.
    metric=pd.read_csv(base/'metrics.csv')
    for row in metric.itertuples():
        subset=saved.loc[(saved.ticker==row.ticker)&(saved.model==row.model)&(saved.seed==row.seed)]
        assert len(subset)==row.n
        assert np.isclose(qlike(subset.actual_variance,subset.predicted_variance).mean(),row.qlike)
(ROOT/'outputs/result_verification.json').write_text(json.dumps(checks,indent=2))
print('Saved model reloads, training scalers, all targets, QLIKE and GARCH analytic forecasts verified.')
