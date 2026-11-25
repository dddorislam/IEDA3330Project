"""Apply the saved historical models to one observed origin; no refitting."""
import argparse,json
import numpy as np
import pandas as pd
import torch
from run_experiment import ROOT,C,Net,prepare

parser=argparse.ArgumentParser()
parser.add_argument('--date',default='2025-12-31',help='Forecast origin within the cached history; not a live quote')
args=parser.parse_args()
rows=[]
for h in C['horizons']:
    for ticker in C['tickers']:
        d=prepare(ticker,h)
        i=d.index.get_loc(pd.Timestamp(args.date))
        variances=[]
        for seed in C['seeds']:
            path=ROOT/f'outputs/h{h}/models/{ticker}_core_seed{seed}'
            meta=json.loads(path.with_suffix('.json').read_text()); hp=meta['hp'];sc=meta['scaler']
            if pd.Timestamp(args.date)<=pd.Timestamp(meta['training_last_label']):
                raise ValueError('Origin is within fitted label history; use an out-of-sample date.')
            a=d[sc['features']].iloc[i-hp['lookback']+1:i+1].to_numpy(float)
            if len(a)!=hp['lookback'] or not np.isfinite(a).all():raise ValueError('Insufficient causal input history')
            x=torch.tensor((a-np.array(sc['mean']))/np.array(sc['std']),dtype=torch.float32).unsqueeze(0)
            net=Net(len(sc['features']),hp['hidden'],hp['dropout'])
            net.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True));net.eval()
            with torch.no_grad():variances.append(float(net(x).item()*sc['variance_scale']))
        v=float(np.mean(variances))
        rows.append(dict(ticker=ticker,origin_date=args.date,horizon_sessions=h,predicted_average_daily_variance=v,predicted_annualized_volatility_pct=100*np.sqrt(252*v),note='Historical-origin forecast using frozen pre-2024 parameters; not current market data.'))
print(json.dumps(rows,indent=2))
