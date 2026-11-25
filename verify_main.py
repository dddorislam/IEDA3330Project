"""Audit the main forecasts and export common-date scores per fund and seed."""
import json
import torch
import numpy as np
import pandas as pd
from main_experiment import OUT,core,frame,scheduled,SEEDS,DIRECT,HYBRID,Model
from write_main_report import metrics

def audit():
    choices=json.loads((OUT/'lock.json').read_text())
    rows=[]
    forecasts=[]
    for stage in ['explore','confirm']:
        for h in [5,22]:
            pick=choices[str(h)]
            reference=None
            for key in dict.fromkeys(['reference',pick['direct'],pick['hybrid'],'HAR','linear','constant']):
                f,_,_=metrics(stage,h,key)
                if reference is None: reference=f.actual_variance
                pd.testing.assert_series_equal(reference,f.actual_variance)
                assert np.isfinite(f.loss).all() and (f.predicted_variance>0).all()
                for ticker,sub in f.groupby(level='ticker'):
                    d=frame(ticker,h,stage)
                    dt=pd.to_datetime(sub.index.get_level_values('origin_date'))
                    np.testing.assert_allclose(sub.actual_variance,[np.mean(d['return'].iloc[d.index.get_loc(t)+1:d.index.get_loc(t)+1+h]**2) for t in dt])
                    if key=='HAR':
                        for t in dt[::50]:
                            past=d.loc[:t].copy()
                            past.loc[past.label_end>t,'target']=np.nan
                            np.testing.assert_allclose(scheduled(past)[-1],d.at[t,'b'])
                    ledger=sub.reset_index().copy()
                    ledger['stage']=stage
                    ledger['ensemble']=key
                    ledger['seed']=-1 if key not in ['HAR','linear','constant'] else 0
                    ledger['issued_har']=d.loc[dt,'b'].values
                    ledger['multiplier']=ledger.predicted_variance/ledger.issued_har
                    month_end=pd.Series(d.index,index=d.index).groupby(d.index.to_period('M')).max()
                    ledger['har_fit_cutoff']=[month_end[t.to_period('M')-1] for t in dt]
                    if key.startswith('hybrid'):
                        assert ledger.multiplier.between(.25-1e-6,4+1e-6).all()
                    forecasts.append(ledger)
                    rows.append(dict(stage=stage,horizon=h,ticker=ticker,model=key,seed=int(ledger.seed.iloc[0]),qlike=sub.loss.mean(),mae=sub.err.abs().mean(),rmse=np.sqrt(np.mean(sub.err**2))))
                    if key in ['HAR','linear','constant']: continue
                    for s in SEEDS:
                        single=pd.read_csv(OUT/stage/f'h{h}'/ticker/'1'/f'{key}_{s}.csv')
                        if s==17:
                            base=OUT/stage/f'h{h}'/ticker/'1'/f'{key}_{s}'
                            meta=json.loads(base.with_suffix('.json').read_text())['scaler']
                            size=DIRECT.get(key,(22,16,False))[0]
                            pos=d.index.get_loc(dt[0])
                            x=d[meta['features']].iloc[pos-size+1:pos+1].to_numpy()
                            x=torch.tensor((x-meta['mean'])/meta['std'],dtype=torch.float32)[None]
                            b=torch.tensor([d.b.iloc[pos]],dtype=torch.float64)
                            model=Model(key,x.shape[-1]).eval()
                            model.load_state_dict(torch.load(base.with_suffix('.pt'),weights_only=True))
                            with torch.no_grad(): fitted=model(x,b)[0].item()
                            if key not in HYBRID: fitted*=meta['variance_scale']
                            np.testing.assert_allclose(fitted,single.predicted_variance.iloc[0],rtol=2e-5)
                        err=100*np.sqrt(252)*(np.sqrt(single.predicted_variance)-np.sqrt(single.actual_variance))
                        rows.append(dict(stage=stage,horizon=h,ticker=ticker,model=key,seed=s,qlike=core.qlike(single.actual_variance,single.predicted_variance).mean(),mae=err.abs().mean(),rmse=np.sqrt(np.mean(err**2))))
    pd.concat(forecasts,ignore_index=True).to_csv(OUT/'forecast_ledger.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'scores_by_fund_and_seed.csv',index=False)
    print('Passed direct target, causal baseline, positivity, multiplier and common-date checks.')

if __name__=='__main__': audit()
