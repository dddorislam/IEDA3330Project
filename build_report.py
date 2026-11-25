"""Render saved empirical results to a compact, reproducible PDF and Markdown report."""
from pathlib import Path
import json, platform, importlib.metadata, sys, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; REPORT=ROOT/'report'; REPORT.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
C=json.loads((ROOT/'config.json').read_text())
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.titlesize':11})
main_models=['Historical5','Historical22','EWMA','GARCH','HAR-style','LSTM-ensemble','LSTM-ablation-ensemble']
names={'Historical5':'Past 5 days','Historical22':'Past 22 days','LSTM-ensemble':'LSTM ensemble','LSTM-ablation-ensemble':'LSTM without range/volume','HAR-style':'HAR-style'}
data={}; metrics={}; macros={}; boots={}; sels={}
for h in C['horizons']:
    base=OUT/f'h{h}'
    data[h]=pd.read_csv(base/'test_predictions.csv',parse_dates=['origin_date'])
    metrics[h]=pd.read_csv(base/'metrics.csv')
    macros[h]=pd.read_csv(base/'macro_metrics.csv')
    boots[h]=pd.read_csv(base/'block_bootstrap.csv')
    sels[h]=json.loads((base/'selection.json').read_text())
    fig,axes=plt.subplots(3,1,figsize=(10,7.2),sharex=True)
    for ax,ticker in zip(axes,C['tickers']):
        d=data[h].loc[data[h].ticker==ticker]
        truth=d.loc[d.model=='LSTM-ensemble']
        ax.plot(truth.origin_date,100*np.sqrt(252*truth.actual_variance),color='#b2b7bf',lw=1,label='Observed forward proxy')
        for model,col in [('GARCH','#de8d2d'),('LSTM-ensemble','#166b91')]:
            g=d.loc[d.model==model]
            ax.plot(g.origin_date,100*np.sqrt(252*g.predicted_variance),color=col,lw=1,label=names.get(model,model))
        ax.set_title(ticker,loc='left'); ax.set_ylabel('Annualized vol (%)'); ax.grid(alpha=.18)
    axes[0].legend(ncol=3,fontsize=8,loc='upper left')
    fig.suptitle(f'{h}-day forecasts | untouched 2024-2025 test',fontweight='bold')
    fig.tight_layout(); fig.savefig(FIG/f'forecasts_h{h}.png',bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,ticker in zip(axes,C['tickers']):
        g=pd.read_csv(base/'calibration.csv'); g=g.loc[g.ticker==ticker]
        x=100*np.sqrt(252*g.mean_predicted_variance); y=100*np.sqrt(252*g.mean_actual_variance)
        lim=max(x.max(),y.max())*1.1
        ax.plot([0,lim],[0,lim],ls='--',color='gray'); ax.scatter(x,y,color='#166b91')
        ax.set_title(ticker); ax.set_xlabel('sqrt(mean predicted variance), %'); ax.set_ylabel('sqrt(mean observed variance), %'); ax.grid(alpha=.15)
    fig.suptitle(f'{h}-day LSTM calibration | five forecast bins'); fig.tight_layout(); fig.savefig(FIG/f'calibration_h{h}.png',bbox_inches='tight'); plt.close(fig)
    hp=sels[h]['hp']; config=f"L{hp['lookback']}_H{hp['hidden']}_D{hp['dropout']}"
    fig,axes=plt.subplots(3,3,figsize=(11,8))
    for row,ticker in enumerate(C['tickers']):
        for col,fold in enumerate([1,2,3]):
            hist=pd.read_json(base/'validation'/f'{ticker}_fold{fold}_{config}_seed17.history.json')
            ax=axes[row,col]
            ax.plot(hist.epoch,hist.train_loss,label='Training');ax.plot(hist.epoch,hist.validation_loss,label='Validation')
            ax.set_title(f'{ticker} / fold {fold}');ax.set_xlabel('Epoch');ax.set_ylabel('Scaled QLIKE');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.suptitle(f'{h}-day selected LSTM | seed 17 learning curves')
    fig.tight_layout();fig.savefig(FIG/f'learning_curves_h{h}.png',bbox_inches='tight');plt.close(fig)

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#17354a'),spaceAfter=18))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))
story=[]
def p(text,style='BodyCustom'): story.append(Paragraph(text,styles[style]))
def heading(text): p(text,'Heading1')
def table(headers,rows,widths=None):
    items=[[Paragraph(escape(str(x)),styles['SmallCustom']) for x in row] for row in [headers]+rows]
    t=Table(items,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#5d7c90')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t); story.append(Spacer(1,10))
def pic(name,w=490):
    from PIL import Image as PILImage
    with PILImage.open(FIG/name) as im: height=w*im.height/im.width
    story.append(Image(str(FIG/name),width=w,height=height))
def newpage(): story.append(PageBreak())
def macro(h): return macros[h].loc[macros[h].model.isin(main_models)].sort_values('qlike')

winner=macro(5).iloc[0]
primary=macro(5).loc[lambda d:d.model=='LSTM-ensemble'].iloc[0]
bestbase=macro(5).loc[lambda d:~d.model.str.startswith('LSTM')].iloc[0]
ci=boots[5].loc[(boots[5].ticker=='Equal-weight ETFs')&(boots[5].baseline==bestbase.model)&(boots[5].block_length==20)].iloc[0]
conclusion=f"The best five-day model by equal-weight mean QLIKE was {names.get(winner.model,winner.model)}. The LSTM ensemble's QLIKE difference versus the strongest baseline ({names.get(bestbase.model,bestbase.model)}) was {primary.qlike-bestbase.qlike:+.4f}; its 20-session block-bootstrap 95% interval was [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]. Negative differences favor the LSTM."

p('Forecasting ETF volatility','TitleCustom')
p('An executed LSTM experiment for IEDA 3330','Heading2')
p('SPY | QQQ | IWM &nbsp;&nbsp; / &nbsp;&nbsp; Daily data 2004-2025 &nbsp;&nbsp; / &nbsp;&nbsp; 7 November 2026','SmallCustom')
heading('1. Introduction and main finding')
p('Can a compact recurrent neural network improve forecasts of ETF risk beyond standard volatility models? This experiment forecasts the next five trading days of average daily squared returns, with a separately selected 22-day robustness experiment. It studies volatility, not price direction or trading profitability.')
p(conclusion)
table(['Five-day model','Mean QLIKE','Vol MAE (pp)','Vol RMSE (pp)'],[[names.get(r.model,r.model),f'{r.qlike:.4f}',f'{r.vol_mae_pp:.2f}',f'{r.vol_rmse_pp:.2f}'] for r in macro(5).itertuples()],[200,90,100,100])
p('Lower QLIKE, MAE and RMSE are better. QLIKE uses decimal daily variance and can be negative. The volatility metrics are equal-weight means of ETF-level errors in annualized percentage points. These are historical out-of-sample results for three selected surviving ETFs.','SmallCustom')
heading('What was completed')
p('Price acquisition and quality checks; causal features and forward labels; three expanding validation folds; eight LSTM configurations per horizon; three seeds for the selected design; an input ablation; five baselines; frozen-parameter test forecasts; synchronized block-bootstrap intervals; saved weights, scalers, forecasts and reproducibility records.')
newpage()

heading('2. Data and methodology')
audit=pd.read_csv(ROOT/'data/audit.csv')
table(['ETF','Observations','Coverage','Missing sessions'],[[r.ticker,r.rows,f'{r.start} to {r.end}',r.missing_sessions] for r in audit.itertuples()],[50,85,260,95])
p('Source: Yahoo Finance through yfinance, downloaded 7 November 2026. Raw snapshots and SHA-256 hashes are preserved. The 2004 observations provide warm-up; model training origins begin in 2005. Prices are checked against the XNYS trading-session calendar, used for the common US equity session dates. Exchange identifiers are retained in source metadata.')
p('The project-folder ETF CSV is a cross-sectional product snapshot; its stock CSV contains individual equities. Neither was substituted for ETF price history. Shiller monthly/annual series and tokenized-asset aggregates were excluded from the daily forecasting model.')
p('Define r[t] = log(adjusted_close[t] / adjusted_close[t-1]). The target v[t,h] is the mean of r[t+1]^2 through r[t+h]^2. Displayed volatility is sqrt(252*v). This zero-mean close-to-close proxy is noisy and is not high-frequency realized variance. Unknown final h labels are dropped.')
p('Inputs: daily return, absolute return, log trailing 5/22/60-session variance, log high/low range, and relative log volume. The ablation removes range and volume. Features use only information available by the forecast origin; scalers and target normalization are fitted on training rows. Forecasts are issued after end-of-day data become available.')
table(['Stage','Training','Evaluation'],[['Fold 1','2005-2016','2017-2018'],['Fold 2','2005-2018','2019-2020'],['Fold 3','2005-2020','2021-2023'],['Final test','2005-2023','2024-2025']],[110,180,200])
p('Training labels must end strictly before the next evaluation block starts; evaluation labels must end within their block. All models share the same evaluable dates. Parameters remain frozen in the final test while observed price histories update. COVID-2020 is validation data, not final test data.')
p('Corporate-action spot checks: the SPY December 2024 dividend agrees with issuer disclosure to the Yahoo export rounding (1.966 versus 1.965548 USD), and the IWM 2-for-1 split date of 9 June 2005 agrees with an SEC-filed description. Yahoo Close history is already split-adjusted; Adj Close adds distribution adjustments. No second split adjustment was applied. This is a spot-check, not a full independent vendor reconciliation.','SmallCustom')
newpage()

heading('3. Models and validation')
p('LSTM: one recurrent layer, last hidden state, dropout, a 16-unit ReLU dense layer, then positive softplus variance output. Training uses Adam (0.001), batch size 32, gradient norm cap 1, maximum 100 epochs and patience 10. The loss is log(predicted variance) + observed variance / predicted variance, implemented after training-only variance normalization.')
p('Architecture search: lookback {22,60}, hidden units {16,32}, dropout {0,0.1}; mean validation QLIKE across folds and ETFs selects one common specification per horizon using seed 17. The selected architecture is then validated with seeds 42 and 123. Each final ETF/seed fit uses its median best epoch from validation. The ensemble averages the three variance forecasts. The ablation uses the selected core architecture with independently validated epoch counts.')
table(['Horizon','Lookback','Hidden units','Dropout'],[[h,sels[h]['hp']['lookback'],sels[h]['hp']['hidden'],sels[h]['hp']['dropout']] for h in C['horizons']],[110,125,130,125])
p('Baselines: past-5 and past-22-session variance; EWMA with decay 0.94; zero-mean Gaussian GARCH(1,1), fitted on percent returns and converted back to decimal variance; ordinary least-squares HAR-style regression on current squared return and trailing 5/22-session variance. Negative HAR-style forecasts are floored at 1e-8. GARCH forecasts average the h conditional future variances.')
p('HAR-style denotes an adaptation of Corsi (2009), not an exact HAR-RV replication. LSTM motivation follows Hochreiter and Schmidhuber (1997); Patton (2011) motivates loss-based comparison with imperfect volatility proxies. No claim is made that an LSTM must outperform simpler models.')
for h in C['horizons']:
    rank=pd.read_csv(OUT/f'h{h}/validation_ranking.csv')
    table([f'{h}-day validation: top 3 of 8 configurations','Mean QLIKE'],[[r.config,f'{r.qlike:.4f}'] for r in rank.head(3).itertuples()],[360,130])
p('Full rankings and epoch-by-epoch learning curves are saved with the experiment outputs.','SmallCustom')
newpage()

heading('4. Five-day test findings')
rows=[]
for ticker in C['tickers']:
    g=metrics[5].loc[(metrics[5].ticker==ticker)&metrics[5].model.isin(['Historical22','EWMA','GARCH','HAR-style','LSTM-ensemble'])]
    for r in g.sort_values('qlike').itertuples(): rows.append([ticker,names.get(r.model,r.model),r.n,f'{r.qlike:.4f}',f'{r.vol_rmse_pp:.2f}',f'{r.oos_r2:.3f}'])
table(['ETF','Model','N','QLIKE','RMSE pp','OOS R2'],rows,[40,160,40,85,80,85])
p('OOS R2 compares variance squared error against the past-22-session baseline on the same dates. A negative value means the model is worse on this metric. Differences between QLIKE and squared-error rankings are possible: they weight forecast errors differently.','SmallCustom')
p(conclusion)
p('Ablation results measure the effect of removing range and volume together, conditional on the core-selected architecture. They do not isolate the contribution of each input or prove causality. No VIX, pooled-asset model, or portfolio strategy was added after observing test results.')
newpage()
heading('5. Forecast paths and calibration')
pic('forecasts_h5.png')
pic('calibration_h5.png')
p('The gray series is the future outcome aligned to its forecast origin. Calibration points compare square roots of mean variances within forecast bins; they are descriptive test diagnostics. Abrupt shocks can remain difficult to anticipate even when volatility persistence is captured.','SmallCustom')
newpage()

heading('6. Uncertainty, seed variation and robustness')
p('Adjacent forward labels overlap. We use 2,000 moving-block bootstrap resamples and resample the same date blocks across ETFs. The intervals quantify sampling uncertainty in the saved loss differences; they do not incorporate all training or model-selection uncertainty. No multiple-comparison adjustment is applied.')
table(['Five-day baseline','LSTM minus baseline','95% interval, block 20'],[[names.get(r.baseline,r.baseline),f'{r.mean_loss_difference:+.4f}',f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]'] for r in boots[5].loc[(boots[5].ticker=='Equal-weight ETFs')&(boots[5].block_length==20)].itertuples()],[205,115,170])
p('Negative differences favor the LSTM; an interval crossing zero does not establish a clear improvement. Companion CSVs include blocks 10 and 40, per-ETF intervals, annual breakdowns, and non-overlapping origins beginning at the first eligible test origin.','SmallCustom')
seedrows=[]
for h in C['horizons']:
    for r in macros[h].loc[macros[h].model=='LSTM'].itertuples():seedrows.append([h,r.seed,f'{r.qlike:.4f}',f'{r.vol_rmse_pp:.2f}'])
table(['Horizon','Seed','Mean QLIKE','Vol RMSE pp'],seedrows,[100,100,145,145])
heading('22-day robustness')
table(['Model','Mean QLIKE','Vol MAE pp','Vol RMSE pp'],[[names.get(r.model,r.model),f'{r.qlike:.4f}',f'{r.vol_mae_pp:.2f}',f'{r.vol_rmse_pp:.2f}'] for r in macro(22).itertuples()],[200,90,100,100])
p(f"The best 22-day model among the declared baselines and ensembles is {names.get(macro(22).iloc[0].model,macro(22).iloc[0].model)} by mean QLIKE. Individual-seed outcomes are reported separately rather than choosing a winning seed after testing.",'SmallCustom')
p('The 22-day model has its own target, boundary purging, validation search and selected epoch counts. Its absolute QLIKE value should not be used to declare it better than a five-day model. Blocks 44/66/88 are reported for this longer horizon.','SmallCustom')
newpage()
heading('7. Longer-horizon forecast paths')
pic('forecasts_h22.png')
pic('calibration_h22.png')
newpage()

heading('8. Conclusion and limitations')
p(conclusion)
p('This study answers a limited empirical question about SPY, QQQ and IWM in 2024-2025 under a fixed forecasting protocol. It does not establish universal ETF predictability or profitable trading. The daily squared-return proxy is noisy, the ETFs are correlated surviving US equity funds, the final test is short, and historical downloads may reflect vendor revisions. The bootstrap intervals are conditional on the fitted models.')
p('The models were fitted on CPU. Deterministic algorithms and recorded seeds improve reproducibility, but exact floating-point results can still depend on library and hardware versions. Validation, final models, transformations, raw-data hashes and per-origin forecasts are saved. Optional VIX and pooled models remain future research rather than being tuned against this test.')
heading('Reproducibility and verification')
p('The project includes config.json; download_data.py; run_experiment.py; evaluate.py; build_report.py; raw and processed data; validation histories; model weights and scaler JSON; prediction and metric CSVs; bootstrap intervals; and environment version records. Run commands are in README.md. Checks cover forward-label indexing, boundary purging, causal-feature perturbations, common dates, finite positive predictions and trading-calendar completeness.')
heading('References and source links')
refs=[
('Yahoo/yfinance historical data and download controls','https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html'),
('Hochreiter & Schmidhuber (1997), Long Short-Term Memory','https://www.bioinf.jku.at/publications/older/2604.pdf'),
('Corsi (2009), A Simple Approximate Long-Memory Model of Realized Volatility','https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf'),
('Patton (2011), Volatility forecast comparison using imperfect volatility proxies','https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf'),
('Rosenbaum & Zhang (2022), On the universality of the volatility formation process','https://arxiv.org/abs/2206.14114'),
('State Street SPY securities report: December 2024 distribution','https://www.ssga.com/library-content/products/fund-docs/etfs/apac/jp/Securities_Report_SPY.pdf'),
('SEC-filed IWM description: June 2005 split','https://www.sec.gov/Archives/edgar/data/19617/000089109210005752/e41298_424b2.pdf')]
for title,url in refs:p(f'<link href="{escape(url)}" color="#166b91">{escape(title)}</link>','SmallCustom')

def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#687989'))
    canvas.drawString(42,25,'IEDA 3330 | ETF volatility forecasting | 7 November 2026')
    canvas.drawRightString(A4[0]-42,25,str(doc.page))

SimpleDocTemplate(str(REPORT/'ETF_volatility_results.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=42,title='ETF volatility forecasting: empirical results',author='IEDA 3330 research project').build(story,onFirstPage=footer,onLaterPages=footer)
lines=['# ETF volatility experiment results','',conclusion,'','## Five-day test: equal-weight means','', '| Model | QLIKE | Volatility MAE (pp) | Volatility RMSE (pp) |','|---|---:|---:|---:|']
for r in macro(5).itertuples(): lines.append(f'| {names.get(r.model,r.model)} | {r.qlike:.4f} | {r.vol_mae_pp:.2f} | {r.vol_rmse_pp:.2f} |')
lines.extend(['','Primary test: 2024-2025; parameters frozen, labels purged at split boundaries. See PDF for methodology, per-ETF findings, uncertainty and 22-day robustness.','','No investment/trading-performance inference was tested.'])
(REPORT/'results_summary.md').write_text('\n'.join(lines),encoding='utf-8')
packages={x.metadata['Name']:x.version for x in importlib.metadata.distributions() if x.metadata['Name']}
(ROOT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),packages=packages,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list(ROOT.glob('*.py'))+[ROOT/'config.json']}),indent=2))
(ROOT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda x:x[0].lower()))+'\n')
print(conclusion)
print('Report saved:',REPORT/'ETF_volatility_results.pdf')
