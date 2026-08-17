"""Supplementary Figure S1: FreeSurfer hippocampal volume by MTA grade (box plots).

Requires outputs of the upstream scripts in the same OUT_DIR:
  analysis/merge_cohort.py   -> cohort_merged.csv
  analysis/benchmark.py      -> oof_seed42_adaptive_ridge_nested.csv
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

d = pd.read_csv(str(OUT_DIR / 'grade_fs_data.csv'))
C_MAN='#555555'; C_E='#0072B2'
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.color':'#E5E5E5','grid.linewidth':0.6,'axes.axisbelow':True})

panels = [('Hippo_Lt','e_Hippo_Lt','ST29SV','A. Left MTA grade vs left hippocampal volume',4,1000),
          ('Hippo_Rt','e_Hippo_Rt','ST88SV','B. Right MTA grade vs right hippocampal volume',4,1000),
          ('man_MTA_bilat','e_MTA_bilat','FS_Hippo_Total','C. Bilateral MTA (L+R) vs total hippocampal volume',8,2000)]

fig, axes = plt.subplots(3,1, figsize=(8.2,10.5))
rng = np.random.RandomState(3)
for ax,(mg,eg,vc,title,maxg,ymin) in zip(axes,panels):
    for g in range(maxg+1):
        for off, col, colr in [(-0.17, mg, C_MAN), (0.17, eg, C_E)]:
            v = d.loc[d[col]==g, vc].values
            if len(v)==0: continue
            bp = ax.boxplot(v, positions=[g+off], widths=0.28, patch_artist=True, showfliers=False,
                            medianprops=dict(color='white',lw=1.4),
                            boxprops=dict(facecolor=colr, edgecolor=colr, alpha=0.85),
                            whiskerprops=dict(color=colr), capprops=dict(color=colr))
            ax.scatter(np.full(len(v), g+off)+rng.uniform(-0.05,0.05,len(v)), v,
                       s=5, color=colr, alpha=0.35, edgecolor='none', zorder=3)
            ax.text(g+off, ymin*0.98 if vc!='FS_Hippo_Total' else 3300, str(len(v)),
                    ha='center', fontsize=6.5, color=colr)
    ax.set_xticks(range(maxg+1)); ax.set_xticklabels(range(maxg+1))
    ax.set_xlabel('MTA grade'); ax.set_ylabel('FreeSurfer volume (mm³)')
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
    if vc=='FS_Hippo_Total': ax.set_ylim(3000, 10600)
    else: ax.set_ylim(950, 5600)
hs=[mlines.Line2D([],[],color=C_MAN,marker='s',ls='None',ms=7,label='Manual CVRS'),
    mlines.Line2D([],[],color=C_E,marker='s',ls='None',ms=7,label='e-CVRS')]
axes[0].legend(handles=hs, frameon=False, fontsize=8.5, loc='upper right')
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'figures/FigureS1_grade_vs_volume.png'), dpi=300, bbox_inches='tight')
print('figure saved')
