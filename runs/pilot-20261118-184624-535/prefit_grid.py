from concurrent.futures import ProcessPoolExecutor
from itertools import product
import time
import run_experiment as r

def prefit(job):
    t,f,l,u,dr=job
    h=22
    hp=dict(lookback=l,hidden=u,dropout=dr)
    path=r.OUT/'h22'/'validation'/f'{t}_fold{f}_{r.tag(hp)}_seed17.json'
    if path.exists(): return
    d=r.prepare(t,h)
    start,end=r.C['folds'][f-1]
    tr=r.indices(d,start,end,True)
    ev=r.indices(d,start,end)
    begin=time.perf_counter()
    p,_,_,history,epoch=r.fit(d,tr,ev,hp,17,r.FEATURES)
    r.save_json(path.with_suffix('.history.json'),history)
    r.records(d,ev,p,t,h,'LSTM',17,f).to_csv(path.with_suffix('.csv'),index=False)
    r.save_json(path,dict(ticker=t,fold=f,hp=hp,seed=17,best_epoch=epoch,qlike=float(r.qlike(d.target.iloc[ev],p).mean()),train_n=len(tr),validation_n=len(ev),seconds=time.perf_counter()-begin))

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(prefit,product(r.C['tickers'],range(1,4),[22,60],[16,32],[0.,.1])))
