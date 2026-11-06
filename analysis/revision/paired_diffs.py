"""Paired bootstrap differences between measures (R2 major 8 / statistical comments).
Seed-42 out-of-fold grades of the published model. 2000 paired resamples of subjects."""
import sys, json
REPO = sys.argv[1]
sys.path.insert(0, REPO + '/analysis'); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import benchmark as B
from ecvrs.stats.regression import fit_ols, compare_models

OUT = B.OUT_DIR
df = pd.read_csv(OUT / 'cohort_merged.csv')
e = pd.read_csv(OUT / 'oof_seed42_full_quadratic_ridge.csv')
df['e_sum'] = e['e_sum'].values
df['negFS'] = -df['FS_Hippo_Total']
NB = 2000
rng = np.random.RandomState(42)


def F_of(sub, oc, col):
    t = sub[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier', oc, col]].dropna()
    cov = t[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier']].values
    z = ((t[col] - t[col].mean()) / t[col].std()).values.reshape(-1, 1)
    return compare_models(fit_ols(cov, t[oc].values), fit_ols(np.hstack([cov, z]), t[oc].values))[0]


def summarize(point, boots):
    b = np.asarray(boots)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return dict(diff=float(point), lo=float(lo), hi=float(hi), p_boot=float(min(p, 1.0)))


res = {}
csf = df.dropna(subset=['ABETA', 'PTAU']).reset_index(drop=True)
# CSF: compare magnitudes in the atrophy direction (more atrophy ~ lower Abeta42, higher p-tau)
for bm, sign in [('ABETA', -1), ('PTAU', 1)]:
    pt = sign * (spearmanr(csf.e_sum, csf[bm])[0] - spearmanr(csf.Manual_Atrophy_Sum, csf[bm])[0])
    bs = []
    for _ in range(NB):
        i = rng.randint(0, len(csf), len(csf)); s = csf.iloc[i]
        bs.append(sign * (spearmanr(s.e_sum, s[bm])[0] - spearmanr(s.Manual_Atrophy_Sum, s[bm])[0]))
    res[f'csf_{bm}_ecvrs_minus_manual'] = summarize(pt, bs)
# FreeSurfer concordance
ok = df.dropna(subset=['FS_Hippo_Total']).reset_index(drop=True)
pt = spearmanr(ok.e_sum, ok.negFS)[0] - spearmanr(ok.Manual_Atrophy_Sum, ok.negFS)[0]
bs = []
for _ in range(NB):
    i = rng.randint(0, len(ok), len(ok)); s = ok.iloc[i]
    bs.append(spearmanr(s.e_sum, s.negFS)[0] - spearmanr(s.Manual_Atrophy_Sum, s.negFS)[0])
res['fs_rho_ecvrs_minus_manual'] = summarize(pt, bs)
# Cognition: incremental F differences (e-CVRS minus manual; e-CVRS minus FreeSurfer)
for oc in B.COGS:
    for other in ['Manual_Atrophy_Sum', 'negFS']:
        pt = F_of(df, oc, 'e_sum') - F_of(df, oc, other)
        bs = []
        for _ in range(NB):
            i = rng.randint(0, len(df), len(df)); s = df.iloc[i].reset_index(drop=True)
            bs.append(F_of(s, oc, 'e_sum') - F_of(s, oc, other))
        res[f'F_{oc}_ecvrs_minus_{"manual" if other.startswith("Manual") else "FS"}'] = summarize(pt, bs)
json.dump(res, open(OUT / 'rev_paired_diffs.json', 'w'), indent=1)
for k, v in res.items():
    print(f"{k:42s} diff {v['diff']:+.3f}  95% CI {v['lo']:+.3f} to {v['hi']:+.3f}  p {v['p_boot']:.3f}")
