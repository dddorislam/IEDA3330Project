"""Draw the exploratory and confirmation paired comparisons from saved intervals."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from main_experiment import ROOT,OUT
d=pd.read_csv(OUT/'paired_intervals.csv')
d=d[((d.h==5)&(d.block==20))|((d.h==22)&(d.block==66))]
fig,axes=plt.subplots(1,2,figsize=(8,3),sharex=True,constrained_layout=True)
for ax,(period,part) in zip(axes,d.groupby('stage',sort=False)):
    for i,row in enumerate(part.itertuples()):
        ax.plot([row.low,row.high],[i,i],color='#345770',lw=2)
        ax.scatter(row.delta,i,color='#17394d',s=24)
    ax.axvline(0,color='.5',lw=1,ls='--')
    ax.set_yticks(range(len(part)),[f'{r.h}d {r.candidate} vs {r.ref}' for r in part.itertuples()],fontsize=8)
    ax.invert_yaxis()
    ax.set_title('2024–25 exploratory' if period=='explore' else '2026 confirmation')
    ax.set_xlabel('Difference in mean QLIKE')
    ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'report'/'ETF_LSTM_intervals.pdf')
