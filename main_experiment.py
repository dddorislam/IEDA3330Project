"""Main experiment. Run validate, lock, explore, confirm in order."""
from pathlib import Path
import argparse, json, time, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize, minimize_scalar
import run_experiment as core

ROOT = core.ROOT
OUT = ROOT/'runs'/'main_20261118'
OUT.mkdir(parents=True,exist_ok=True)
G = np.log(4.)
DIRECT = {'reference':(60,32,True),'short':(22,32,True),'scalar':(22,32,False),'narrow':(22,16,False)}
HYBRID = {'hybrid0':0.,'hybrid01':.01,'hybrid1':.1}
SEEDS = [17,42,123]
BLOCKS = dict(explore=('2024-01-01','2025-12-31'),confirm=('2026-01-01','2026-11-06'))

def scheduled(d):
    x = np.column_stack([np.ones(len(d)),d[['q','var5','var22']]])
    b = np.full(len(d),np.nan)
    for _, pos in pd.Series(np.arange(len(d)),index=d.index).groupby(d.index.to_period('M')):
        first = int(pos.iloc[0])
        if d.index[first].year < 2010: continue
        eligible = (d.index.year>=2005) & (d.label_end<d.index[first]) & np.isfinite(x).all(axis=1)
        coeff = np.linalg.lstsq(x[eligible],d.target[eligible],rcond=None)[0]
        b[pos.values] = np.maximum(1e-8,x[pos.values]@coeff)
    return b

def frame(ticker,h,stage):
    if stage=='confirm':
        archived=core.ROOT
        try:
            core.ROOT=OUT/'holdout'
            d=core.prepare(ticker,h)
        finally: core.ROOT=archived
    else: d=core.prepare(ticker,h)
    d['b']=scheduled(d)
    d['logb']=np.log(d.b)
    return d

class Model(nn.Module):
    def __init__(self,key,n):
        super().__init__()
        self.hybrid=key in HYBRID
        _,width,dense=DIRECT.get(key,(22,16,False))
        self.rnn=nn.LSTM(n,width,batch_first=True)
        self.dropout=nn.Dropout(.1)
        self.head=nn.Sequential(nn.Linear(width,16),nn.ReLU(),nn.Linear(16,1)) if dense else nn.Linear(width,1)
        if self.hybrid:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self,x,b):
        z=self.head(self.dropout(self.rnn(x)[1][0][-1])).flatten()
        if self.hybrid:
            g=G*torch.tanh(z.double()/G)
            return (b*g.exp()).clamp_min(1e-8),g
        return nn.functional.softplus(z)+1e-6,torch.zeros_like(z)

def train(d,tr,ev,key,seed,epochs,path):
    hybrid=key in HYBRID
    features=core.FEATURES+(['logb'] if hybrid else [])
    L=DIRECT.get(key,(22,16,False))[0]
    x,y,xv,yv,sc=core.tensors(d,tr,ev,L,features)
    b=torch.tensor(d.b.iloc[tr].to_numpy())
    bv=torch.tensor(d.b.iloc[ev].to_numpy())
    scale=sc['variance_scale']
    if hybrid:
        y=torch.tensor(d.target.iloc[tr].to_numpy())
        yv=torch.tensor(d.target.iloc[ev].to_numpy())
    torch.manual_seed(seed)
    model=Model(key,len(features))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=float('inf')
    history=[]
    start=time.perf_counter()
    for epoch in range(1,(epochs or 100)+1):
        model.train()
        for ix in torch.randperm(len(tr)).split(32):
            p,g=model(x[ix],b[ix])
            loss=(p.log()+y[ix]/p+HYBRID.get(key,0)*g.square()).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        if epochs: continue
        model.eval()
        with torch.no_grad():
            pv=torch.cat([model(a,c)[0] for a,c in zip(xv.split(512),bv.split(512))])
            score=float((pv.log()+yv/pv).mean())
        history.append(score)
        if score<best-1e-5:
            best=score
            best_epoch=epoch
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        elif epoch-best_epoch>=10: break
    if not epochs: model.load_state_dict(state)
    elapsed=time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        tick=time.perf_counter()
        pred=torch.cat([model(a,c)[0] for a,c in zip(xv.split(512),bv.split(512))]).numpy()
        micros=(time.perf_counter()-tick)*1e6/len(ev)
    if not hybrid: pred*=scale
    assert np.isfinite(pred).all() and np.all(pred>0)
    torch.save(model.state_dict(),path.with_suffix('.pt'))
    return pred,dict(epoch=epochs or best_epoch,seconds=elapsed,us=micros,params=sum(p.numel() for p in model.parameters()),scaler=sc,history=history)

def run(job):
    ticker,h,stage=job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d=frame(ticker,h,stage)
    eligible=d.b.notna().rolling(60).sum().eq(60).to_numpy()
    folds=core.C['folds'] if stage=='validate' else [BLOCKS[stage]]
    choices={} if stage=='validate' else json.loads((OUT/'lock.json').read_text())[str(h)]
    for fold,(start,end) in enumerate(folds,1):
        tr=core.indices(d,start,end,True)
        ev=core.indices(d,start,end)
        tr,ev=tr[eligible[tr]],ev[eligible[ev]]
        assert len(ev) and d.label_end.iloc[tr].max()<d.index[ev].min()
        folder=OUT/stage/f'h{h}'/ticker/str(fold)
        folder.mkdir(parents=True,exist_ok=True)
        baseline=d.b.iloc[tr].to_numpy()
        y=d.target.iloc[tr].to_numpy()
        def obj(g):
            p=np.maximum(1e-8,baseline*np.exp(g))
            return core.qlike(y,p).mean()
        k=minimize_scalar(obj,bounds=(-G,G),method='bounded').x
        columns=core.FEATURES+['logb']
        # Linear correction sees the same 22-session sequence as the hybrid.
        x=d[columns].to_numpy()
        rows=np.unique((tr[:,None]-np.arange(22)).flatten())
        x=(x-x[rows].mean(0))/np.maximum(x[rows].std(0),1e-12)
        def design(ix):
            return np.column_stack([np.ones(len(ix)),x[ix[:,None]-np.arange(21,-1,-1)].reshape(len(ix),-1)])
        design_tr=design(tr)
        def linear_loss(w):
            tan=np.tanh(design_tr@w/G)
            p=baseline*np.exp(G*tan)
            grad=(1-y/np.maximum(1e-8,p))*(p>1e-8)*(1-tan*tan)
            return obj(G*tan)+.01*np.dot(w,w),design_tr.T@grad/len(y)+.02*w
        linear=minimize(linear_loss,np.zeros(design_tr.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
        if not linear.success: raise RuntimeError(linear.message)
        for name,g in [('HAR',0),('constant',k),('linear',G*np.tanh(design(ev)@linear.x/G))]:
            pv=np.maximum(1e-8,d.b.iloc[ev].to_numpy()*np.exp(g))
            core.records(d,ev,pv,ticker,h,name,0,fold).to_csv(folder/(name+'.csv'),index=False)
        keys=list(DIRECT)+list(HYBRID) if stage=='validate' else sorted({'reference',choices['direct'],choices['hybrid']})
        for key in keys:
            for seed in SEEDS:
                path=folder/f'{key}_{seed}'
                if path.with_suffix('.json').exists(): continue
                epochs=None if stage=='validate' else choices['epochs'][f'{ticker}_{key}_{seed}']
                pv,info=train(d,tr,ev,key,seed,epochs,path)
                ledger=core.records(d,ev,pv,ticker,h,key,seed,fold)
                ledger['issued_har']=d.b.iloc[ev].to_numpy()
                ledger.to_csv(path.with_suffix('.csv'),index=False)
                path.with_suffix('.json').write_text(json.dumps(info))
            print(stage,h,ticker,fold,key,flush=True)

def lock():
    result={}
    for h in [5,22]:
        scores=[]
        epochs={}
        for key in list(DIRECT)+list(HYBRID):
            for ticker in core.C['tickers']:
                for fold in [1,2,3]:
                    folder=OUT/'validate'/f'h{h}'/ticker/str(fold)
                    files=[folder/f'{key}_{s}.csv' for s in SEEDS]
                    frames=[pd.read_csv(p) for p in files]
                    pred=np.mean([f.predicted_variance for f in frames],axis=0)
                    meta=json.loads(files[0].with_suffix('.json').read_text())
                    scores.append((key,core.qlike(frames[0].actual_variance,pred).mean(),meta['params'],meta['us']))
                for s in SEEDS:
                    epochs[f'{ticker}_{key}_{s}']=int(np.median([json.loads((folder.parent/str(i)/f'{key}_{s}.json').read_text())['epoch'] for i in [1,2,3]]))
        rank=pd.DataFrame(scores,columns=['model','qlike','params','us']).groupby('model').mean()
        rank.to_csv(OUT/f'ranking_h{h}.csv')
        direct=rank.loc[list(DIRECT)]
        selected=direct[direct.qlike<=direct.qlike.min()+.005].sort_values(['params','us']).index[0]
        result[h]=dict(direct=selected,hybrid=rank.loc[list(HYBRID)].qlike.idxmin(),epochs=epochs)
    result['locked_at_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    result['confirmation_period']=BLOCKS['confirm']
    (OUT/'lock.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    stage=sys.argv[1]
    if stage=='lock': lock()
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(run,[(t,h,stage) for h in [5,22] for t in core.C['tickers']]))

