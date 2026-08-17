"""Supplementary S5-S6: manual CVRS & e-CVRS MTA grades versus FreeSurfer hippocampal volumes.

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
from scipy.stats import spearmanr, kruskal, mannwhitneyu

df = pd.read_csv(str(OUT_DIR / 'cohort_merged.csv'))
e42 = pd.read_csv(str(OUT_DIR / 'oof_seed42_adaptive_ridge_nested.csv'))

# per-side FS volumes (earliest scan per RID, same rule as main analysis)
fs = pd.read_csv(str(DATA_DIR / 'UCSF_FS.csv'), low_memory=False)
fs['EXAMDATE_dt'] = pd.to_datetime(fs['EXAMDATE'], errors='coerce')
fs = fs.sort_values('EXAMDATE_dt').drop_duplicates('RID', keep='first')
df = df.merge(fs[['RID','ST29SV','ST88SV']], on='RID', how='left')
assert df.ST29SV.notna().all()
# sanity: sum matches
assert np.allclose(df.ST29SV+df.ST88SV, df.FS_Hippo_Total)

df['man_MTA_bilat'] = df.Hippo_Lt + df.Hippo_Rt
df['e_MTA_bilat'] = e42.Hippo_Lt + e42.Hippo_Rt
df['e_Hippo_Lt'] = e42.Hippo_Lt; df['e_Hippo_Rt'] = e42.Hippo_Rt

out = {}

def rho_ci(a, b, nb=2000, seed=42):
    np.random.seed(seed); a=np.asarray(a); b=np.asarray(b); m=len(a); vals=[]
    for _ in range(nb):
        i=np.random.choice(m,m,replace=True); vals.append(spearmanr(a[i],b[i])[0])
    r,p = spearmanr(a,b)
    return r, p, np.percentile(vals,2.5), np.percentile(vals,97.5)

def paired_rho_diff(g1, g2, vol, nb=2000, seed=42):
    """bootstrap CI for rho(manual,vol)-rho(ecvrs,vol) on same subjects"""
    np.random.seed(seed); m=len(vol); vals=[]
    g1=np.asarray(g1); g2=np.asarray(g2); v=np.asarray(vol)
    for _ in range(nb):
        i=np.random.choice(m,m,replace=True)
        vals.append(spearmanr(g1[i],v[i])[0]-spearmanr(g2[i],v[i])[0])
    d = spearmanr(g1,vol)[0]-spearmanr(g2,vol)[0]
    return d, np.percentile(vals,2.5), np.percentile(vals,97.5)

# ---- S5A: per-grade volume distributions ----
def grade_table(gradecol, volcol, maxg):
    rows=[]
    for g in range(maxg+1):
        v = df.loc[df[gradecol]==g, volcol]
        if len(v)==0: rows.append(dict(grade=g,n=0)); continue
        rows.append(dict(grade=g, n=int(len(v)), mean=float(v.mean()), sd=float(v.std()),
                         median=float(v.median()), q1=float(v.quantile(.25)), q3=float(v.quantile(.75))))
    return rows

configs = [
 ('manual','Hippo_Lt','ST29SV',4,'Left MTA (manual) vs left hippocampal volume'),
 ('ecvrs','e_Hippo_Lt','ST29SV',4,'Left MTA (e-CVRS) vs left hippocampal volume'),
 ('manual','Hippo_Rt','ST88SV',4,'Right MTA (manual) vs right hippocampal volume'),
 ('ecvrs','e_Hippo_Rt','ST88SV',4,'Right MTA (e-CVRS) vs right hippocampal volume'),
 ('manual','man_MTA_bilat','FS_Hippo_Total',8,'Bilateral MTA (manual) vs total hippocampal volume'),
 ('ecvrs','e_MTA_bilat','FS_Hippo_Total',8,'Bilateral MTA (e-CVRS) vs total hippocampal volume'),
]
out['grade_tables'] = {}
out['stats'] = {}
for method, gcol, vcol, maxg, label in configs:
    key = f'{method}_{gcol}'
    out['grade_tables'][key] = dict(label=label, rows=grade_table(gcol, vcol, maxg))
    # trend: spearman of grade vs -volume (atrophy direction) + kruskal-wallis
    r,p,lo,hi = rho_ci(df[gcol], -df[vcol])
    groups = [df.loc[df[gcol]==g, vcol].values for g in sorted(df[gcol].unique()) if (df[gcol]==g).sum()>0]
    H, kw_p = kruskal(*groups)
    # linear slope of volume on grade (mm3 per grade step)
    slope = np.polyfit(df[gcol], df[vcol], 1)[0]
    out['stats'][key] = dict(rho=r, p=p, lo=lo, hi=hi, kruskal_H=float(H), kruskal_p=float(kw_p),
                             slope_mm3_per_grade=float(slope))

# ---- paired differences manual vs e-CVRS ----
out['paired'] = {}
for side, mg, eg, vcol in [('left','Hippo_Lt','e_Hippo_Lt','ST29SV'),
                           ('right','Hippo_Rt','e_Hippo_Rt','ST88SV'),
                           ('bilateral','man_MTA_bilat','e_MTA_bilat','FS_Hippo_Total')]:
    d, lo, hi = paired_rho_diff(df[mg], df[eg], -df[vcol])
    out['paired'][side] = dict(diff=d, lo=lo, hi=hi)

# ---- discrimination: detect lowest-tertile volume ----
from sklearn.metrics import roc_auc_score
out['auc'] = {}
def auc_ci(y, s, nb=2000, seed=42):
    np.random.seed(seed); y=np.asarray(y); s=np.asarray(s); m=len(y); vals=[]
    for _ in range(nb):
        i=np.random.choice(m,m,replace=True)
        if len(np.unique(y[i]))<2: continue
        vals.append(roc_auc_score(y[i],s[i]))
    return roc_auc_score(y,s), np.percentile(vals,2.5), np.percentile(vals,97.5)
for side, mg, eg, vcol in [('left','Hippo_Lt','e_Hippo_Lt','ST29SV'),
                           ('right','Hippo_Rt','e_Hippo_Rt','ST88SV'),
                           ('bilateral','man_MTA_bilat','e_MTA_bilat','FS_Hippo_Total')]:
    low = (df[vcol] <= df[vcol].quantile(1/3)).astype(int)
    a1 = auc_ci(low, df[mg]); a2 = auc_ci(low, df[eg])
    out['auc'][side] = dict(manual=dict(auc=a1[0],lo=a1[1],hi=a1[2]),
                            ecvrs=dict(auc=a2[0],lo=a2[1],hi=a2[2]), n_low=int(low.sum()))

json.dump(out, open(str(OUT_DIR / 'grade_fs_results.json'),'w'), indent=1, default=float)
df[['RID','Hippo_Lt','Hippo_Rt','e_Hippo_Lt','e_Hippo_Rt','man_MTA_bilat','e_MTA_bilat','ST29SV','ST88SV','FS_Hippo_Total']].to_csv(str(OUT_DIR / 'grade_fs_data.csv'), index=False)

# print summary
for key, s in out['stats'].items():
    print(f"{key:28s} rho={s['rho']:.3f} ({s['lo']:.3f},{s['hi']:.3f}) KW p={s['kruskal_p']:.2g} slope={s['slope_mm3_per_grade']:.0f} mm3/grade")
print("\npaired manual-ecvrs rho diff:")
for k,v in out['paired'].items(): print(f"  {k:10s} {v['diff']:+.3f} ({v['lo']:.3f},{v['hi']:.3f})")
print("\nAUC lowest-tertile volume:")
for k,v in out['auc'].items():
    print(f"  {k:10s} manual {v['manual']['auc']:.3f} ({v['manual']['lo']:.3f}-{v['manual']['hi']:.3f}) | e-CVRS {v['ecvrs']['auc']:.3f} ({v['ecvrs']['lo']:.3f}-{v['ecvrs']['hi']:.3f})")
