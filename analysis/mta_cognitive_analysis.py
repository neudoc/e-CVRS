"""Supplementary S7: hierarchical regressions of six cognitive outcomes on bilateral MTA grades.

Requires outputs of the upstream scripts in the same OUT_DIR:
  analysis/merge_cohort.py   -> cohort_merged.csv
  analysis/benchmark.py      -> oof_seed42_adaptive_ridge_nested.csv
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecvrs.stats.regression import fit_ols, compare_models, fdr_correction

df = pd.read_csv(str(OUT_DIR / 'cohort_merged.csv'))
e42 = pd.read_csv(str(OUT_DIR / 'oof_seed42_adaptive_ridge_nested.csv'))
df['man_MTA'] = df.Hippo_Lt + df.Hippo_Rt
df['e_MTA'] = e42.Hippo_Lt + e42.Hippo_Rt

cogs = ['MMSE','ADAS13','CDRSB','ADNI_MEM','ADNI_EF','ADNI_LAN']
out = {}
for oc in cogs:
    sub = df[['AGE','Gender_Dummy','PTEDUCAT','APOE4_Carrier',oc,'man_MTA','e_MTA','FS_Hippo_Total']].dropna()
    cov = sub[['AGE','Gender_Dummy','PTEDUCAT','APOE4_Carrier']].values
    y = sub[oc].values
    m1 = fit_ols(cov, y)
    row = {'N': len(sub), 'cov_adjR2': m1['adj_r_squared']}
    ps=[]
    for lab, col, flip in [('manual_mta','man_MTA',1),('ecvrs_mta','e_MTA',1),('fs','FS_Hippo_Total',-1)]:
        zp = (sub[col]-sub[col].mean())/sub[col].std()*flip
        m2 = fit_ols(np.hstack([cov, zp.values.reshape(-1,1)]), y)
        F,p = compare_models(m1,m2)
        row[lab] = dict(adjR2=m2['adj_r_squared'], F=F, p=p, beta=m2['coef'][-1], se=m2['se'][-1])
        ps.append(p)
    adj = fdr_correction(ps)
    for lab, ap in zip(['manual_mta','ecvrs_mta','fs'], adj): row[lab]['p_bh']=float(ap)
    out[oc]=row

json.dump(out, open(str(OUT_DIR / 'mta_cognitive_results.json'),'w'), indent=1, default=float)
for oc,row in out.items():
    line=f"{oc:9s} N={row['N']} covR2={row['cov_adjR2']:.3f}"
    for lab in ['manual_mta','ecvrs_mta','fs']:
        r=row[lab]
        sig='*' if r['p_bh']<0.05 else ' '
        line+=f" | {lab}: F={r['F']:5.1f} pbh={r['p_bh']:.4f}{sig}"
    print(line)
