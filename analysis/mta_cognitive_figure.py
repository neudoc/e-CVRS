"""Supplementary Figure S2: cognitive associations of bilateral MTA grades (forest plot).

Requires outputs of the upstream scripts in the same OUT_DIR:
  analysis/merge_cohort.py   -> cohort_merged.csv
  analysis/benchmark.py      -> oof_seed42_adaptive_ridge_nested.csv
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

o = json.load(open(OUT_DIR / 'mta_cognitive_results.json'))
df = pd.read_csv(OUT_DIR / 'cohort_merged.csv')
labels = [('MMSE','MMSE'),('ADAS13','ADAS-Cog-13'),('CDRSB','CDR-SB'),
          ('ADNI_MEM','ADNI-MEM'),('ADNI_EF','ADNI-EF'),('ADNI_LAN','ADNI-LAN')]

C_MAN='#555555'; C_E='#0072B2'; C_FS='#D55E00'
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.color':'#E5E5E5','grid.linewidth':0.6,'axes.axisbelow':True})
flip={'MMSE':-1,'ADAS13':1,'CDRSB':1,'ADNI_MEM':-1,'ADNI_EF':-1,'ADNI_LAN':-1}
sds={k:df[k].std() for k,_ in labels}
preds=[('manual_mta','Manual bilateral MTA',C_MAN,'s'),('ecvrs_mta','e-CVRS bilateral MTA',C_E,'o'),('fs','FreeSurfer hippocampus',C_FS,'D')]
fig,ax=plt.subplots(figsize=(7.6,5.6))
yy=0; yticks=[]; ylabs=[]
for k,lab in labels:
    for pk,plab,col,mk in preds:
        r=o[k][pk]
        b=r['beta']*flip[k]/sds[k]; se=r['se']/sds[k]
        sig=r['p_bh']<0.05
        ax.errorbar(b,yy,xerr=1.96*se,fmt=mk,color=col,mfc=col if sig else 'white',capsize=2.5,ms=6,lw=1.2)
        yy-=1
    yticks.append(yy+2); ylabs.append(lab); yy-=0.8
ax.axvline(0,color='#999',lw=0.9)
ax.set_yticks(yticks); ax.set_yticklabels(ylabs)
ax.set_xlabel('Standardized β (95% CI), oriented so positive = more atrophy → worse cognition')
hs=[mlines.Line2D([],[],color=c,marker=mk,ls='None',ms=6,label=l) for pk,l,c,mk in preds]
ax.legend(handles=hs,frameon=False,fontsize=8.5,loc='lower right')
ax.set_title('Bilateral MTA grade and cognition (filled = BH-adjusted p<0.05)',loc='left',fontsize=10,fontweight='bold')
plt.tight_layout(); (OUT_DIR / 'figures').mkdir(exist_ok=True); plt.savefig(OUT_DIR / 'figures' / 'FigureS2_mta_cognition.png',dpi=300,bbox_inches='tight'); plt.close()
print('S7 + Figure S2 built')
