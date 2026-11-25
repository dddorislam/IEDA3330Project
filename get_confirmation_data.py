from main_experiment import OUT,core
import exchange_calendars as xc
import yfinance as yf
import pandas as pd
import hashlib,json
raw=OUT/'holdout'/'data'/'raw'
raw.mkdir(parents=True,exist_ok=True)
yf.set_tz_cache_location(str(OUT/'yahoo_cache'))
expected=xc.get_calendar('XNYS',start='2025-11-01',end='2026-11-20').sessions_in_range('2026-01-01','2026-11-06').tz_localize(None)
manifest={}
for ticker in core.C['tickers']:
    prior=pd.read_csv(core.ROOT/'data'/'raw'/f'{ticker}.csv',index_col='date',parse_dates=True)
    fresh=yf.Ticker(ticker).history(start='2025-12-31',end='2026-11-07',auto_adjust=False,repair=False,actions=True,raise_errors=True)
    fresh.index=fresh.index.tz_localize(None).normalize()
    if not fresh.index[1:].equals(expected) or fresh.index[0]!=prior.index[-1]: raise RuntimeError('Incomplete holdout data')
    fresh['Adj Close']*=prior['Adj Close'].iloc[-1]/fresh['Adj Close'].iloc[0]
    merged=pd.concat([prior,fresh.iloc[1:]])
    path=raw/f'{ticker}.csv'
    merged.to_csv(path,index_label='date')
    manifest[ticker]=dict(sessions=len(expected),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(raw.parent/'manifest.json').write_text(json.dumps(manifest,indent=2))
