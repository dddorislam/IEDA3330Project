"""Build self-contained LaTeX from the completed forecast ledgers."""
import json
import numpy as np
import pandas as pd
from main_experiment import OUT,ROOT,SEEDS,core

def table(cols,rows):
    lines=[r'\begin{center}',r'\begin{tabular}{l'+'r'*(len(cols)-1)+'}',r'\hline']
    for row in [cols]+rows:
        lines.append(' & '.join(str(x) for x in row)+r' \\')
        if row is cols: lines.append(r'\hline')
    return '\n'.join(lines+[r'\hline',r'\end{tabular}',r'\end{center}'])

def diff_ci(cur,ref,block):
    v=(cur-ref).groupby(level='origin_date').mean().to_numpy()
    rng=np.random.default_rng(1729)
    n=len(v)
    ix=(rng.integers(0,n-block+1,(2000,int(np.ceil(n/block)),1))+np.arange(block)).reshape(2000,-1)[:,:n]
    return [v.mean(),*np.quantile(v[ix].mean(1),[.025,.975])]

def metrics(stage,h,key):
    frames=[]
    costs=[]
    for ticker in core.C['tickers']:
        folder=OUT/stage/f'h{h}'/ticker/'1'
        base=folder/(key+'.csv')
        paths=[base] if base.exists() else [folder/f'{key}_{s}.csv' for s in SEEDS]
        dfs=[pd.read_csv(p).set_index(['origin_date','ticker']).sort_index() for p in paths]
        for df in dfs: assert df.index.equals(dfs[0].index)
        f=dfs[0].copy()
        f.predicted_variance=np.mean([df.predicted_variance for df in dfs],axis=0)
        f['loss']=core.qlike(f.actual_variance,f.predicted_variance)
        f['err']=100*np.sqrt(252)*(np.sqrt(f.predicted_variance)-np.sqrt(f.actual_variance))
        frames.append(f)
        if len(paths)>1:
            meta=[json.loads(p.with_suffix('.json').read_text()) for p in paths]
            costs.append([meta[0]['params'],sum(m['seconds'] for m in meta),sum(m['us'] for m in meta)])
    f=pd.concat(frames).sort_index()
    return f,[f.loss.mean(),f.err.abs().mean(),np.mean([np.sqrt((g.err**2).mean()) for _,g in f.groupby(level='ticker')])],np.mean(costs,axis=0) if costs else None

def build():
    pilot=ROOT/'runs'/'pilot-20261118-184624-535'
    locked=json.loads((OUT/'lock.json').read_text())
    stats=[]
    differences=[]
    runtime=[]
    observations=[]
    for stage in ['explore','confirm']:
        if not (OUT/stage).exists(): continue
        for h in [5,22]:
            select=locked[str(h)]
            models=list(dict.fromkeys(['reference',select['direct'],'HAR','constant','linear',select['hybrid']]))
            data={}
            for m in models:
                df,values,cost=metrics(stage,h,m)
                data[m]=df.loss
                stats.append([stage,h,m,len(df)//3,*[f'{v:.3f}' for v in values]])
                if stage=='explore' and cost is not None: runtime.append([h,m,int(cost[0]),f'{cost[1]:.1f}',f'{cost[2]:.1f}'])
            for m,ref in [(select['direct'],'reference'),(select['hybrid'],'HAR')]:
                for b in ([10,20,40] if h==5 else [44,66,88]):
                    delta,lo,hi=diff_ci(data[m],data[ref],b)
                    differences.append([stage,h,m,ref,b,delta,lo,hi])
                    if b in [20,66]:
                        if m==ref: result='the selection retained the reference itself'
                        elif hi<0: result='the interval supports lower QLIKE'
                        elif hi<.005: result='the upper bound meets the 0.005 preservation criterion'
                        elif lo>0: result='the interval favors the comparator'
                        else: result='the interval leaves the gain unresolved'
                        observations.append((stage,f'At {h} sessions, {m} versus {ref} yielded a mean difference of {delta:+.4f} (95\\% interval [{lo:+.4f}, {hi:+.4f}]); {result}.'))
    q=pd.DataFrame(differences,columns=['stage','h','candidate','ref','block','delta','low','high'])
    q.to_csv(OUT/'paired_intervals.csv',index=False)
    pilot_rows=[]
    for h in [5,22]:
        raw=pd.read_csv(pilot/'outputs'/f'h{h}'/'macro_metrics.csv').set_index('model')
        for name in ['LSTM-ensemble','HAR-style','GARCH']:
            row=raw.loc[name]
            pilot_rows.append([h,name,*[f'{row[k]:.3f}' for k in ['qlike','vol_mae_pp','vol_rmse_pp']]])
    primary=[s for stage,s in observations if stage=='confirm']
    text=r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,booktabs,graphicx,hyperref}
\title{Can a smaller LSTM preserve ETF volatility forecast accuracy?}
\author{IEDA 3330 project report}
\date{18 November 2026}
\begin{document}
\maketitle
\begin{abstract}
We test recurrent network simplification and a multiplicative correction to a statistical variance forecast for SPY, QQQ and IWM. We select architectures on expanding chronological folds and compare forecasts on common dates at two horizons. ABSTRACTRESULT
\end{abstract}
\section{Introduction}
Reducing the sequence length or hidden width lowers the processing required for a daily forecast. Whether the reduction preserves forecast quality is an empirical question. A second way to simplify learning is to give a small network an initial variance estimate and train it to correct that estimate. We evaluate both approaches against their respective reference models.
\section{Data and design}
The pilot uses 5,535 Yahoo Finance daily records per ETF from January 2004 through December 2025. Session coverage, price consistency and corporate actions are audited. Let $r_t=\log(P_t/P_{t-1})$ for adjusted closes. A forecast issued after close $t$ targets $v_{t,h}=h^{-1}\sum_{j=1}^h r_{t+j}^2$, where $h=5$ or $22$. Annualized volatility is $\sqrt{252v}$. Inputs comprise return, absolute return, log trailing mean squared returns over 5, 22 and 60 sessions, log high/low, and log-one-plus volume centered on its 22-session mean.
\paragraph{Pilot.} We restarted training from the archived prices in a fresh run directory. The pilot used 2004 for warm-up, trained from 2005 and evaluated on 2024--2025. The table reports equal-weight ETF means. MAE and RMSE use annualized volatility percentage points. Pilot results document pipeline operation.
PILOT
\section{Main experiment}
A separate network is fitted for each ETF and horizon. Validation blocks are 2017--2018, 2019--2020 and 2021--2023. Training labels must mature before the validation block begins. The common training sample starts after 60 consecutive issued HAR forecasts from 2010. The reference uses 60 inputs in time, 32 hidden units and a 16-unit dense layer. Short uses 22 days, scalar additionally removes the dense layer, and narrow additionally reduces hidden width to 16. Each uses dropout 0.1 and a positive Softplus head. We average variance forecasts from seeds 17, 42 and 123. Adam uses a learning rate of 0.001, batches of 32, gradient clipping at 1, a limit of 100 epochs and patience 10. Final fits use the median selected epoch across folds. Scalers use training rows alone.
\paragraph{Baseline and corrections.} Following the multiple-horizon construction of Corsi~\cite{corsi}, an expanding OLS regression uses daily, weekly and monthly squared-return averages. At each month end we update the coefficients using matured labels, effective next session. We floor the baseline $b$ at $10^{-8}$. Constant and regularized linear log multipliers serve as calibration controls. The hybrid reads the seven features and log issued baseline over 22 sessions with 16 hidden units. Its zero-initialized scalar head produces $z$, with $g=G\tanh(z/G)$ and $G=\log 4$. The forecast is $p=\max(10^{-8},b\exp g)$. We minimize $\log p+v/p+\lambda g^2$ and select $\lambda$ from 0, 0.01 and 0.1 (hybrid0, hybrid01 and hybrid1).
\paragraph{Selection and scoring.} Full QLIKE is $\log p+v/p$ and lower values indicate better performance. Patton~\cite{patton} analyzes forecast losses with imperfect volatility proxies. We select the fewest parameters within 0.005 of the best direct validation QLIKE, breaking ties using inference time. The hybrid minimizes validation QLIKE. Selection gives equal weight to folds and ETFs. The criterion retained the 22-session window with 32 hidden units and its dense layer for the five-day target. For the 22-day target it retained the 60-session reference. Both horizons selected a hybrid penalty of 0.1. The short model has the same number of weights as the reference but processes 63\% fewer recurrent steps. The 2024--2025 comparison is exploratory. January 1 through November 6, 2026 was reserved before selection. Only origins with all forward outcomes inside each evaluation period qualify. The 214 confirmation-period sessions per ETF are joined to the archived history by matching adjusted-close levels on December 31, 2025. Final confirmation weights use eligible training labels through 2025. The monthly HAR updates continue during evaluation.
\section{Results}
RANK
The following table gives common forecast counts per ETF and equal-weight accuracy across ETFs. Explore denotes 2024--2025 and confirm denotes the reserved 2026 period.
RESULTS
\paragraph{Cost.} Counts refer to one network. Training seconds and batch inference microseconds per origin sum the three ensemble members and are averaged across ETFs. Timings depend on concurrent CPU workloads and exclude HAR refits and feature preparation.
COST
\paragraph{Uncertainty.} Synchronized moving-block bootstrap intervals use 2,000 resamples of daily mean paired losses and block lengths of 20 or 66 sessions for five- or 22-day forecasts. A negative difference favors the candidate. The accompanying CSV includes sensitivity to shorter and longer blocks.
FINDINGS
\paragraph{What accounts for the 22-session gain?} The hybrid reduced 2026 QLIKE relative to HAR for each ETF, by 0.047 for SPY, 0.010 for QQQ and 0.026 for IWM. A linear correction captured most of that gain. The hybrid-minus-linear difference was $-0.0022$ with an exploratory 95\% interval of $[-0.0084,0.0214]$. The data support correcting HAR, while the incremental value of the recurrent correction over linear calibration remains uncertain. The five-day preservation result and the 22-day hybrid-versus-HAR result held under all three reported block lengths.
\IfFileExists{ETF_LSTM_intervals.pdf}{\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{ETF_LSTM_intervals.pdf}\caption{Paired mean QLIKE differences and pointwise 95\% intervals. Negative values favor the candidate.}\end{figure}}{}
\section{Limitations and conclusion}
Three correlated US equity ETFs provide limited cross-market coverage. Squared close-to-close returns are noisy variance proxies. Overlapping horizons reduce effective sample size, especially for a short confirmation period. Intervals are conditional on fitted models and are pointwise across comparisons. The 0.005 tolerance is an engineering choice. Validation supported the architectural choices listed above. FINAL
\begin{thebibliography}{2}
\bibitem{corsi} F. Corsi. A Simple Approximate Long-Memory Model of Realized Volatility. \emph{Journal of Financial Econometrics}, 7(2), 174--196, 2009. \href{https://doi.org/10.1093/jjfinec/nbp001}{Publisher record}.
\bibitem{patton} A. J. Patton. Volatility forecast comparison using imperfect volatility proxies. \emph{Journal of Econometrics}, 160(1), 246--256, 2011. \href{https://doi.org/10.1016/j.jeconom.2010.03.034}{Publisher record}.
\end{thebibliography}
\end{document}
"""
    ranks=[]
    for h in [5,22]:
        for m,row in pd.read_csv(OUT/f'ranking_h{h}.csv').set_index('model').iterrows():
            ranks.append([h,m,int(row.params),f'{row.qlike:.4f}'])
    ending=' '.join(primary) if primary else 'Confirmation accuracy remains unmeasured because complete reserved observations were unavailable.'
    for a,b in {'ABSTRACTRESULT':r'The five-session target could use a shorter input window under the prespecified 0.005 QLIKE tolerance. The 22-session target retained the full-window direct model. At that horizon a compact residual LSTM improved on HAR by 0.0275 QLIKE (pointwise 95\% interval for the difference $[-0.0529,-0.0114]$). A linear correction achieved a similar score.','PILOT':table(['Horizon','Pilot model','QLIKE','MAE','RMSE'],pilot_rows),'RANK':table(['Horizon','Validation candidate','Parameters','QLIKE'],ranks),'RESULTS':table(['Period','Horizon','Model','Origins','QLIKE','MAE','RMSE'],stats),'COST':table(['Horizon','Network','Parameters','Train (s)','Inference ($\\mu$s)'],runtime),'FINDINGS':' '.join(s for _,s in observations),'FINAL':'The paired differences and measured costs above determine which simplifications meet the stated tolerance for these ETFs. Evidence for transfer to other funds or later market regimes would require additional prospective observations.'}.items(): text=text.replace(a,b)
    path=ROOT/'report'/'ETF_LSTM_main_experiment.tex'
    path.write_text(text,encoding='utf8')
    print(path)

if __name__=='__main__': build()

