"""Reproducible, resumable ETF experiment. Run --stage validate before --stage test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse, json, itertools, copy, random, time, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from arch import arch_model

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT/'config.json').read_text())
OUT = ROOT/'outputs'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
FEATURES = ['return','abs_return','log_var5','log_var22','log_var60','range','relative_volume']

def save_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str))

def prepare(ticker, h):
    d = pd.read_csv(ROOT/f'data/raw/{ticker}.csv',index_col='date',parse_dates=True)
    d['return'] = np.log(d['Adj Close']).diff()
    d['q'] = d['return']**2
    d['abs_return'] = d['return'].abs()
    for k in [5,22,60]:
        d[f'var{k}'] = d.q.rolling(k).mean()
        d[f'log_var{k}'] = np.log(d[f'var{k}']+1e-8)
    d['range'] = np.log(d.High/d.Low)
    v = np.log1p(d.Volume)
    d['relative_volume'] = v-v.rolling(22).mean()
    d['target'] = d.q.rolling(h).mean().shift(-h)
    d['label_end'] = pd.Series(d.index,index=d.index).shift(-h)
    d['ewma'] = d.q.ewm(alpha=.06,adjust=False).mean()
    return d

def indices(d, start, end, train=False):
    good = d[FEATURES].notna().all(axis=1).rolling(60).sum().eq(60)
    good &= d.target.notna() & (d.index>=C['train_start'])
    if train:
        good &= (d.index < start) & (d.label_end < pd.Timestamp(start))
    else:
        good &= (d.index>=start) & (d.index<=end) & (d.label_end<=pd.Timestamp(end))
    return np.flatnonzero(good.to_numpy())

def tensors(d, tr, ev, lookback, features):
    # Unique feature rows occurring in eligible training windows only.
    rows = np.unique(np.concatenate([np.arange(i-lookback+1,i+1) for i in tr]))
    a = d[features].to_numpy(float)
    mean = a[rows].mean(axis=0)
    std = a[rows].std(axis=0)
    std = np.where(std<1e-12,1.,std)
    scaled = (a-mean)/std
    def windows(ii):
        return torch.tensor(np.stack([scaled[i-lookback+1:i+1] for i in ii]),dtype=torch.float32)
    c = float(d.target.iloc[tr].mean())
    return windows(tr), torch.tensor(d.target.iloc[tr].to_numpy()/c,dtype=torch.float32), windows(ev), torch.tensor(d.target.iloc[ev].to_numpy()/c,dtype=torch.float32), dict(mean=mean.tolist(),std=std.tolist(),variance_scale=c,features=features)

class Net(nn.Module):
    def __init__(self,f,hidden,dropout):
        super().__init__()
        self.lstm=nn.LSTM(f,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden,16),nn.ReLU(),nn.Linear(16,1),nn.Softplus())
    def forward(self,x):
        _,(hidden,_) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)+1e-6

def qlike(y,p):
    return np.log(p)+y/p

def fit(d,tr,ev,hp,seed,features,fixed_epochs=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x,y,xv,yv,scaler=tensors(d,tr,ev,hp['lookback'],features)
    model=Net(len(features),hp['hidden'],hp['dropout'])
    opt=torch.optim.Adam(model.parameters(),lr=C['learning_rate'])
    best=float('inf'); state=None; stale=0; best_epoch=0; history=[]
    for epoch in range(1,(fixed_epochs or C['max_epochs'])+1):
        model.train(); order=torch.randperm(len(x)); losses=[]
        for idx in order.split(C['batch_size']):
            opt.zero_grad(); pred=model(x[idx]); loss=(pred.log()+y[idx]/pred).mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            losses.append(float(loss.detach())*len(idx))
        model.eval()
        with torch.no_grad():
            if fixed_epochs is None:
                pv=torch.cat([model(b) for b in xv.split(512)])
                vl=float((pv.log()+yv/pv).mean())
            else: vl=None  # Never consult test labels to stop training.
        history.append(dict(epoch=epoch,train_loss=sum(losses)/len(x),validation_loss=vl))
        if fixed_epochs is None:
            if vl<best-1e-5:
                best=vl; best_epoch=epoch; state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            if stale>=C['patience']: break
    if fixed_epochs is None: model.load_state_dict(state)
    else: best_epoch=fixed_epochs
    model.eval()
    with torch.no_grad(): p=torch.cat([model(b) for b in xv.split(512)]).numpy()*scaler['variance_scale']
    assert np.isfinite(p).all() and (p>0).all()
    return p,model,scaler,history,best_epoch

def records(d,ev,p,ticker,h,model,seed,fold):
    return pd.DataFrame(dict(ticker=ticker,origin_date=d.index[ev],label_end_date=d.label_end.iloc[ev].to_numpy(),horizon=h,model=model,seed=seed,fold=fold,actual_variance=d.target.iloc[ev].to_numpy(),predicted_variance=p))

def baseline(d,tr,ev,h):
    outputs={f'Historical{k}':d[f'var{k}'].iloc[ev].to_numpy() for k in [5,22]}
    outputs['EWMA']=d.ewma.iloc[ev].to_numpy()
    c=d.target.iloc[tr].mean()
    a=np.column_stack([np.ones(len(d)),d[['q','var5','var22']].to_numpy()/c])
    coef=np.linalg.lstsq(a[tr],d.target.iloc[tr]/c,rcond=None)[0]
    outputs['HAR-style']=np.maximum(a[ev]@coef*c,1e-8)
    # Fit parameters only to training returns; filter causally through evaluation.
    r=(100*d['return']).dropna()
    eligible=r.loc[(r.index>=C['train_start']) & (r.index<=d.index[tr[-1]])]
    fitres=arch_model(eligible,mean='Zero',vol='GARCH',p=1,q=1,rescale=False).fit(disp='off',show_warning=False)
    assert fitres.convergence_flag==0, 'GARCH optimization failed'
    omega,alpha,beta=[float(fitres.params[k]) for k in ['omega','alpha[1]','beta[1]']]
    variance=float(fitres.conditional_volatility.iloc[-1]**2)
    last_return=float(eligible.iloc[-1])
    forecasts={}
    # Start immediately after last fitted observation. Variance is conditional at current day.
    for date,ret in r.loc[r.index>d.index[tr[-1]]].items():
        variance=omega+alpha*last_return**2+beta*variance
        nextvar=omega+alpha*float(ret)**2+beta*variance
        path=[nextvar]
        for _ in range(1,h): path.append(omega+(alpha+beta)*path[-1])
        forecasts[date]=np.mean(path)/10000
        last_return=float(ret)
    outputs['GARCH']=np.array([forecasts[date] for date in d.index[ev]])
    return outputs,dict(garch=dict(omega=omega,alpha=alpha,beta=beta),har_coefficients=coef.tolist(),har_scale=float(c))

def tag(hp): return f"L{hp['lookback']}_H{hp['hidden']}_D{hp['dropout']}"

def validation(h):
    base=OUT/f'h{h}'; base.mkdir(exist_ok=True)
    grid=[dict(lookback=l,hidden=u,dropout=dr) for l,u,dr in itertools.product(C['lookbacks'],C['hidden_units'],C['dropouts'])]
    scores=[]
    for hp in grid:
        for ticker in C['tickers']:
            d=prepare(ticker,h)
            for fold,(start,end) in enumerate(C['folds'],1):
                path=base/'validation'/f'{ticker}_fold{fold}_{tag(hp)}_seed17.json'
                if path.exists(): result=json.loads(path.read_text())
                else:
                    tr=indices(d,start,end,True); ev=indices(d,start,end)
                    assert d.label_end.iloc[tr].max()<d.index[ev].min()
                    begin=time.time()
                    p,model,scaler,hist,epoch=fit(d,tr,ev,hp,17,FEATURES)
                    result=dict(ticker=ticker,fold=fold,hp=hp,seed=17,best_epoch=epoch,qlike=float(qlike(d.target.iloc[ev].to_numpy(),p).mean()),train_n=len(tr),validation_n=len(ev),seconds=time.time()-begin)
                    save_json(path,result); save_json(path.with_suffix('.history.json'),hist)
                    records(d,ev,p,ticker,h,'LSTM',17,fold).to_csv(path.with_suffix('.csv'),index=False)
                    print('VALIDATION',h,result,flush=True)
                scores.append(result)
    rank=pd.DataFrame([dict(config=tag(x['hp']),qlike=x['qlike']) for x in scores]).groupby('config').qlike.mean().sort_values()
    rank.to_csv(base/'validation_ranking.csv')
    selected=next(hp for hp in grid if tag(hp)==rank.index[0])
    # Additional seeds only for selected architecture, and predeclared ablation.
    selected_results=[]
    for variant,features in [('core',FEATURES),('ablation',FEATURES[:5])]:
        for ticker in C['tickers']:
            d=prepare(ticker,h)
            for fold,(start,end) in enumerate(C['folds'],1):
                tr=indices(d,start,end,True); ev=indices(d,start,end)
                bp=base/'validation'/f'{ticker}_fold{fold}_baselines.csv'
                if not bp.exists():
                    preds,pars=baseline(d,tr,ev,h)
                    pd.concat([records(d,ev,p,ticker,h,name,0,fold) for name,p in preds.items()]).to_csv(bp,index=False)
                    save_json(bp.with_suffix('.json'),pars)
                for seed in C['seeds']:
                    path=base/'validation'/f'{ticker}_fold{fold}_{tag(selected)}_seed{seed}{"_ablation" if variant=="ablation" else ""}.json'
                    if path.exists(): result=json.loads(path.read_text())
                    else:
                        p,model,scaler,hist,epoch=fit(d,tr,ev,selected,seed,features)
                        result=dict(ticker=ticker,fold=fold,hp=selected,seed=seed,best_epoch=epoch,qlike=float(qlike(d.target.iloc[ev].to_numpy(),p).mean()))
                        save_json(path,result); save_json(path.with_suffix('.history.json'),hist)
                        records(d,ev,p,ticker,h,'LSTM' if variant=='core' else 'LSTM-ablation',seed,fold).to_csv(path.with_suffix('.csv'),index=False)
                        print('SELECTED VALIDATION',h,variant,result,flush=True)
                    selected_results.append(dict(**result,variant=variant))
    save_json(base/'selection.json',dict(horizon=h,hp=selected,validation_results=selected_results,config_sha256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()))
    print('SELECTION LOCKED',h,selected,flush=True)

def test(h):
    base=OUT/f'h{h}'; selection=json.loads((base/'selection.json').read_text()); hp=selection['hp']
    assert selection['config_sha256']==hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()
    all_records=[]
    start,end=C['test']
    for ticker in C['tickers']:
        d=prepare(ticker,h); tr=indices(d,start,end,True); ev=indices(d,start,end)
        assert d.label_end.iloc[tr].max()<d.index[ev].min()
        processed=ROOT/'data/processed'; processed.mkdir(exist_ok=True)
        d.to_csv(processed/f'{ticker}_h{h}.csv')
        preds,pars=baseline(d,tr,ev,h)
        save_json(base/f'{ticker}_baseline_parameters.json',pars)
        all_records.extend(records(d,ev,p,ticker,h,name,0,'test') for name,p in preds.items())
        for variant,features in [('core',FEATURES),('ablation',FEATURES[:5])]:
            ensemble=[]
            for seed in C['seeds']:
                epochs=int(np.median([x['best_epoch'] for x in selection['validation_results'] if x['ticker']==ticker and x['seed']==seed and x['variant']==variant]))
                path=base/'models'/f'{ticker}_{variant}_seed{seed}'
                predpath=path.with_suffix('.csv')
                if predpath.exists(): rec=pd.read_csv(predpath); p=rec.predicted_variance.to_numpy()
                else:
                    p,model,scaler,hist,_=fit(d,tr,ev,hp,seed,features,fixed_epochs=epochs)
                    path.parent.mkdir(exist_ok=True)
                    torch.save(model.state_dict(),path.with_suffix('.pt'))
                    save_json(path.with_suffix('.json'),dict(hp=hp,seed=seed,epochs=epochs,scaler=scaler,training_last_origin=str(d.index[tr[-1]]),training_last_label=str(d.label_end.iloc[tr].max()),horizon=h))
                    save_json(path.with_suffix('.history.json'),hist)
                    rec=records(d,ev,p,ticker,h,'LSTM' if variant=='core' else 'LSTM-ablation',seed,'test')
                    rec.to_csv(predpath,index=False)
                    print('FINAL FIT',h,ticker,variant,seed,'epochs',epochs,flush=True)
                ensemble.append(p); all_records.append(rec)
            all_records.append(records(d,ev,np.mean(ensemble,axis=0),ticker,h,'LSTM-ensemble' if variant=='core' else 'LSTM-ablation-ensemble',-1,'test'))
    pd.concat(all_records,ignore_index=True).to_csv(base/'test_predictions.csv',index=False)
    print('TEST COMPLETE',h,flush=True)

def checks():
    for h in C['horizons']:
        for ticker in C['tickers']:
            d=prepare(ticker,h)
            for i in [200,1500,4000]:
                assert np.isclose(d.target.iloc[i],d.q.iloc[i+1:i+h+1].mean(),rtol=1e-10)
                assert d.label_end.iloc[i]==d.index[i+h]
            assert d.target.tail(h).isna().all()
            for start,end in C['folds']+[C['test']]:
                tr=indices(d,start,end,True); ev=indices(d,start,end)
                assert len(tr)>1000 and len(ev)>400
                assert d.label_end.iloc[tr].max()<d.index[ev].min()
                assert d.label_end.iloc[ev].max()<=pd.Timestamp(end)
            # Perturb future prices and independently rebuild causal core features.
            cutoff=3000; alt=d.copy(); alt.loc[alt.index>alt.index[cutoff],'Adj Close']*=1.5
            r=np.log(alt['Adj Close']).diff()
            assert np.allclose(r.iloc[1:cutoff+1],d['return'].iloc[1:cutoff+1])
            for k in [5,22,60]:
                assert np.allclose(np.log(r.pow(2).rolling(k).mean()+1e-8).iloc[k:cutoff+1],d[f'log_var{k}'].iloc[k:cutoff+1])
    save_json(OUT/'verification.json',dict(forward_labels=True,purged_splits=True,future_perturbation_causal_features=True,checked_at=pd.Timestamp.now().isoformat()))
    print('All label, split and causal-feature checks passed.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['check','validate','test'],required=True); parser.add_argument('--horizon',type=int,default=5); args=parser.parse_args()
    if args.stage=='check': checks()
    elif args.stage=='validate': validation(args.horizon)
    else: test(args.horizon)
