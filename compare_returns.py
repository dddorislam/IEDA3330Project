"""Two return experiments using the archived daily prices and forecast ledger."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import run_experiment as core

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'main_20261118'
OUT=ROOT/'runs'/'return_comparison'
OUT.mkdir(exist_ok=True)
TICKERS=['SPY','QQQ','IWM']
SEEDS=[17,42,123]

def windows(d,tr,ev,L):
    a=d[core.FEATURES].to_numpy()
    lag=np.arange(1-L,1)
    rows=a[np.unique(tr[:,None]+lag)]
    a=(a-rows.mean(0))/rows.std(0).clip(1e-10)
    return [torch.tensor(a[ix[:,None]+lag],dtype=torch.float32) for ix in [tr,ev]]

class Net(nn.Module):
    def __init__(self,width,dense):
        super().__init__()
        self.rnn=nn.LSTM(7,width,batch_first=True)
        self.head=nn.Sequential(nn.Linear(width,16),nn.ReLU(),nn.Linear(16,1)) if dense else nn.Linear(width,1)
    def forward(self,x):
        return self.head(self.rnn(x)[1][0][-1]).flatten()

def fit(x,y,xv,yv,seed,width,epochs=None):
    torch.manual_seed(seed)
    model=Net(width,width==32)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    mu,scale=y.mean(),y.std()
    train_y=torch.tensor((y-mu)/scale,dtype=torch.float32)
    best=np.inf
    for e in range(1,(epochs or 15)+1):
        for idx in torch.randperm(len(x)).split(128):
            pred=model(x[idx]); loss=(pred-train_y[idx]).square().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        if epochs is None:
            with torch.no_grad(): score=np.mean((model(xv).numpy()*scale+mu-yv)**2)
            if score<best: best,chosen=score,e
            elif e-chosen>=4:break
    if epochs is None:return chosen
    with torch.no_grad(): return model(xv).numpy()*scale+mu

def direct(job):
    ticker,h=job
    torch.set_num_threads(1)
    core.ROOT=RUN/'holdout'
    d=core.prepare(ticker,h)
    price=d['Adj Close']
    d['target']=price.shift(-h)/price-1
    tr=core.indices(d,'2021-01-01','2023-12-31',True)
    val=core.indices(d,'2021-01-01','2023-12-31')
    final=core.indices(d,'2026-01-01','2026-11-06',True)
    ev=core.indices(d,'2026-01-01','2026-11-06')
    assert d.label_end.iloc[tr].max()<d.index[val[0]] and d.label_end.iloc[final].max()<d.index[ev[0]]
    y=d.target.to_numpy()
    ledger=pd.DataFrame(dict(date=d.index[ev],label_end=d.label_end.iloc[ev].values,ticker=ticker,h=h,actual=y[ev],zero=0.,mean=y[final].mean()))
    metadata={}
    for L,W,name in [(22,16,'compact'),(60,32,'reference')]:
        x,xv=windows(d,tr,val,L); xf,xe=windows(d,final,ev,L)
        if L==22:
            a=xf[:,-1,:].numpy().astype(float); center=a.mean(0); a-=center
            coef=np.linalg.solve(a.T@a+.01*len(a)*np.eye(7),a.T@(y[final]-y[final].mean()))
            ledger['ridge']=y[final].mean()+(xe[:,-1,:].numpy()-center)@coef
        preds=[]
        for seed in SEEDS:
            epoch=fit(x,y[tr],xv,y[val],seed,W)
            preds.append(fit(xf,y[final],xe,None,seed,W,epoch))
            metadata[f'{name}_{seed}']=epoch
        ledger[name]=np.mean(preds,axis=0)
    assert np.isfinite(ledger.select_dtypes('number')).all().all()
    ledger.to_csv(OUT/f'{ticker}_{h}.csv',index=False)
    (OUT/f'{ticker}_{h}_epochs.json').write_text(json.dumps(metadata))
    print(ticker,h,len(ev),flush=True)

def portfolio_weights(variance):
    return np.minimum(1, .10 / np.sqrt(252 * variance)) / len(TICKERS)


def policies():
    ledger=pd.read_csv(RUN/'forecast_ledger.csv',parse_dates=['origin_date','label_end_date'])
    calibration=ledger.query("stage == 'explore' and ensemble == 'HAR'").copy()
    assert calibration.origin_date.min() >= pd.Timestamp('2024-01-01')
    assert calibration.label_end_date.max() < pd.Timestamp('2026-01-01')
    calibration['weight']=portfolio_weights(calibration.predicted_variance)
    fixed=calibration.groupby(['horizon','ticker']).weight.mean().unstack().loc[:,TICKERS]
    assert np.isfinite(fixed).all().all()
    fixed.to_csv(OUT/'fixed_weights.csv')
    d=ledger.query("stage=='confirm'")
    dates=sorted(set.intersection(*[set(g.origin_date) for _,g in d.groupby(['horizon','ensemble','ticker'])]))
    prices=pd.DataFrame({t:pd.read_csv(RUN/'holdout/data/raw'/f'{t}.csv',parse_dates=['date']).set_index('date')['Adj Close'] for t in TICKERS})
    i=prices.index.get_indexer(dates)
    assert i.min()>=0 and i.max()+2<len(prices) and np.all(np.diff(i)==1)
    forward=prices.iloc[i+2].to_numpy()/prices.iloc[i+1].to_numpy()-1
    logs=[]
    for (h,m),g in d.groupby(['horizon','ensemble']):
        var=g.pivot(index='origin_date',columns='ticker',values='predicted_variance').loc[dates,TICKERS].to_numpy()
        weights=portfolio_weights(var)
        targets=[(m,weights)]
        if m=='HAR':
            targets.append(('fully invested',np.ones_like(weights)/len(TICKERS)))
            targets.append(('fixed exposure',np.broadcast_to(fixed.loc[h].to_numpy(),weights.shape)))
        for policy,w in targets:
            previous=np.zeros(3)
            for j,target in enumerate(w):
                turnover=np.abs(target-previous).sum()
                cost=.0001*turnover
                net=target@forward[j]-cost
                previous=target*(1+forward[j])/(1+net)
                logs.append([h,policy,dates[j],prices.index[i[j]+1],prices.index[i[j]+2],net,target.sum(),turnover])
    d=pd.DataFrame(logs,columns=['h','model','signal','entry','exit','net','exposure','turnover'])
    d.to_csv(OUT/'daily_strategies.csv',index=False)
    stats=[]
    for (h,m),g in d.groupby(['h','model']):
        stats.append([h,m,len(g),g.net.mean()*252,(1+g.net).prod()-1,g.net.std()*np.sqrt(252),g.exposure.mean()])
    pd.DataFrame(stats,columns=['h','model','n','annual_mean','period_return','annual_vol','mean_invested']).to_csv(OUT/'strategy_summary.csv',index=False)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policies-only', action='store_true', help='Replay portfolios using saved forecasts.')
    args=parser.parse_args()
    policies()
    if not args.policies_only:
        with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(direct,[(t,h) for h in [5,22] for t in TICKERS]))
