"""Precompute independent pilot fits. Publish the cache marker after both outputs."""
import itertools,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import run_experiment as r

def prefit(job):
    h,t,f,v,s,hp=job
    if v=='core' and s==17: return
    folder=r.OUT/f'h{h}'/'validation'
    k=r.tag(hp)
    name=f'{t}_fold{f}_{k}_seed{s}'+('_ablation' if v=='ablation' else '')
    path=folder/(name+'.json')
    if path.exists(): return
    d=r.prepare(t,h)
    start,end=r.C['folds'][f-1]
    tr=r.indices(d,start,end,True)
    ev=r.indices(d,start,end)
    p,_,_,hist,epoch=r.fit(d,tr,ev,hp,s,r.FEATURES if v=='core' else r.FEATURES[:5])
    if path.exists(): return
    result=dict(ticker=t,fold=f,hp=hp,seed=s,best_epoch=epoch,qlike=float(r.qlike(d.target.iloc[ev].to_numpy(),p).mean()))
    for suffix,content in [('.history.json',json.dumps(hist)),('.csv',r.records(d,ev,p,t,h,'LSTM' if v=='core' else 'LSTM-ablation',s,f).to_csv(index=False)),('.json',json.dumps(result))]:
        final=folder/(name+suffix)
        tmp=folder/(name+suffix+'.partial')
        tmp.write_text(content)
        tmp.replace(final)

def jobs(h):
    folder=r.OUT/f'h{h}'/'validation'
    while True:
        fits=[json.loads(p.read_text()) for p in folder.glob('*_seed17.json')]
        if len(fits)>=72: break
        time.sleep(5)
    grouped={}
    for x in fits: grouped.setdefault(r.tag(x['hp']),[]).append(x['qlike'])
    best=min(grouped,key=lambda tag:np.mean(grouped[tag]))
    hp=next(x['hp'] for x in fits if r.tag(x['hp'])==best)
    return [(h,t,f,v,s,hp) for v,t,f,s in itertools.product(['ablation','core'],reversed(r.C['tickers']),[3,2,1],r.C['seeds'])]

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        for h in [5,22]:
            list(pool.map(prefit,jobs(h)))
