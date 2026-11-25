"""Align existing forecasts by fund, origin, label end and horizon. No refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent
run=ROOT/'runs'/'return_comparison'
OUT=ROOT/'report'/'risk_return'
OUT.mkdir(parents=True,exist_ok=True)
funds=['SPY','QQQ','IWM']
key=['ticker','date','label_end','h']
ret=pd.concat([pd.read_csv(run/f'{t}_{h}.csv') for t in funds for h in [5,22]])
vol=pd.read_csv(ROOT/'runs/main_20261118/forecast_ledger.csv').query("stage=='confirm'").rename(columns={'origin_date':'date','label_end_date':'label_end','horizon':'h'})
panel=ret.copy()
for source in ['HAR','reference','hybrid1']:
    sub=vol[vol.ensemble==source][key+['predicted_variance']].rename(columns={'predicted_variance':source+'_var'})
    panel=panel.merge(sub,on=key,how='outer',validate='one_to_one',indicator=True)
    assert panel.pop('_merge').eq('both').all()
    panel[source+'_risk']=np.sqrt(panel.h*panel[source+'_var'])
observed=vol.loc[vol.ensemble=='HAR',key+['actual_variance']]
panel=panel.merge(observed,on=key,validate='one_to_one',how='left')
assert panel.actual_variance.notna().all()
panel['realized_risk']=np.sqrt(panel.h*panel.actual_variance)
panel.to_csv(OUT/'paired_forecasts.csv',index=False)
buckets=[];assoc=[];quality=[]
for h,g in panel.groupby('h'):
    assert g.groupby('ticker').size().eq(209 if h==5 else 192).all()
    g=g.copy()
    g['band']=g.groupby('ticker').hybrid1_risk.transform(lambda x:pd.qcut(x,3,labels=False))
    for band,f in g.groupby('band'):
        return_rmse=np.sqrt(((f.reference-f.actual)**2).mean())
        buckets.append(dict(h=h,band=int(band),n=len(f),return_rmse=return_rmse,**f[['hybrid1_risk','realized_risk','reference','ridge','compact','actual']].mean().to_dict()))
    for m in ['mean','ridge','compact','reference','zero']:
        err=g[m]-g.actual
        quality.append(dict(h=h,model=m,expected=g[m].mean(),rmse=np.sqrt((err**2).mean()),ratio=(g[m]/g.hybrid1_risk).mean()))
    for source in ['hybrid1','HAR','reference']:
        for m in ['reference','compact','ridge','actual']:
            a=g.pivot(index='date',columns='ticker',values=source+'_risk').values
            b=g.pivot(index='date',columns='ticker',values=m).values
            def corr(a,b):
                a=rankdata(a,axis=1);b=rankdata(b,axis=1)
                a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
                return ((a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))).mean(-1)
            val=corr(a[None],b[None])[0]
            lo=hi=np.nan
            if source=='hybrid1':
                n=len(a);block=20 if h==5 else 66
                rng=np.random.default_rng(23)
                vals=[]
                for j in range(20):
                    ix=(rng.integers(n-block+1,size=(100,int(np.ceil(n/block)),1))+np.arange(block)).reshape(100,-1)[:,:n]
                    vals.extend(corr(a[ix],b[ix]))
                lo,hi=np.quantile(vals,[.025,.975])
            assoc.append(dict(h=h,risk_model=source,return_series=m,rho=val,lo=lo,hi=hi))
for name,rows in [('risk_bands',buckets),('associations',assoc),('forecast_quality',quality)]:
    d=pd.DataFrame(rows);d.to_csv(OUT/f'{name}.csv',index=False);print(name,d.round(4).to_string(index=False))
