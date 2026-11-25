"""Download immutable Yahoo snapshots; audit before running models."""
from pathlib import Path
import json, hashlib, platform, importlib.metadata
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf
import exchange_calendars as xc

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT/'config.json').read_text())
raw = ROOT/'data/raw'
raw.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(ROOT/'data/yahoo_cache'))
manifest = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'sources': []}
calendar = xc.get_calendar('XNYS', start='2003-12-01', end='2026-01-15')
expected = calendar.sessions_in_range(cfg['download_start'], '2025-12-31').tz_localize(None)
audits = []
for ticker in cfg['tickers']:
    path = raw/f'{ticker}.csv'
    if not path.exists():
        obj = yf.Ticker(ticker)
        df = obj.history(start=cfg['download_start'], end=cfg['download_end_exclusive'], auto_adjust=False, actions=True, repair=False, raise_errors=True)
        if df.empty:
            raise RuntimeError(f'Empty download: {ticker}')
        df.index = df.index.tz_convert('America/New_York').tz_localize(None).normalize()
        df.index.name = 'date'
        df.to_csv(path)
        (raw/f'{ticker}_metadata.json').write_text(json.dumps(obj.history_metadata, default=str, indent=2))
    df = pd.read_csv(path, index_col='date', parse_dates=True)
    required = ['Open','High','Low','Close','Adj Close','Volume']
    assert not df.index.duplicated().any()
    assert df.index.is_monotonic_increasing
    assert not df[required].isna().any().any()
    assert (df[['Open','High','Low','Close','Adj Close']] > 0).all().all()
    assert (df.Volume >= 0).all()
    bad = (df.High < df[['Open','Low','Close']].max(axis=1)-1e-6) | (df.Low > df[['Open','High','Close']].min(axis=1)+1e-6)
    assert not bad.any(), f'{ticker} OHLC inconsistency'
    missing = expected.difference(df.index)
    extra = df.index.difference(expected)
    assert len(missing)==0 and len(extra)==0, f'{ticker}: missing {missing}, extra {extra}'
    rets = np.log(df['Adj Close']).diff()
    events = df.loc[(df['Dividends'] != 0) | (df['Stock Splits'] != 0)].copy()
    events['adjusted_return'] = rets.reindex(events.index)
    events.to_csv(raw/f'{ticker}_actions_audit.csv')
    audits.append(dict(ticker=ticker, rows=len(df), start=str(df.index.min().date()), end=str(df.index.max().date()), missing_sessions=len(missing), duplicate_dates=int(df.index.duplicated().sum()), dividends=int((df.Dividends!=0).sum()), splits=int((df['Stock Splits']!=0).sum()), max_abs_log_return=float(rets.abs().max())))
    manifest['sources'].append(dict(ticker=ticker,url=f'https://finance.yahoo.com/quote/{ticker}/history/',file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(), adjustment='Yahoo Adj Close for returns; unadjusted OHLC for within-day range',exchange_calendar='XNYS',currency='USD',timezone='America/New_York'))
    print(ticker, audits[-1], flush=True)
manifest['packages'] = {p: importlib.metadata.version(p) for p in ['yfinance','pandas','numpy','exchange_calendars']}
(ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2))
pd.DataFrame(audits).to_csv(ROOT/'data/audit.csv',index=False)
print('All data quality gates passed.',flush=True)
