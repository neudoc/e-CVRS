"""Main-text tables for the published e-CVRS analysis.

Reads the out-of-fold grades produced by analysis/benchmark.py for the winning
calibration (full_quadratic_ridge) and computes every number reported in
Tables 1-5:

  Table 1  cohort description
  Table 2  agreement with blinded expert ratings (kappa, ICC, Bland-Altman)
  Table 3  covariate-adjusted association with six cognitive outcomes
  Table 4  association with CSF biomarkers
  Table 5  calibration-candidate benchmark and composite ranks

Writes final_results.json into ECVRS_OUT_DIR and prints a headline summary.
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecvrs.stats.agreement import cohen_kappa_quadratic, intraclass_correlation
from ecvrs.stats.regression import fit_ols, compare_models, fdr_correction

WINNER = 'full_quadratic_ridge'
NB = 2000                      # bootstrap replicates
LOBES = [('Hippo_Lt', 5), ('Hippo_Rt', 5), ('Front', 4), ('Parietal', 4), ('Temporal', 4)]
COGS = ['MMSE', 'ADAS13', 'CDRSB', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN']

df = pd.read_csv(OUT_DIR / 'cohort_merged.csv')
e42 = pd.read_csv(OUT_DIR / f'oof_seed42_{WINNER}.csv')
res_all = pd.read_csv(OUT_DIR / 'benchmark_results.csv')
out = {}

# ---------------------------------------------------------------- Table 1 ---
t1 = {'N': len(df)}
t1['age_mean'], t1['age_sd'] = df.AGE.mean(), df.AGE.std()
t1['female_pct'] = 100 * df.Gender_Dummy.mean()
t1['educ_mean'], t1['educ_sd'] = df.PTEDUCAT.mean(), df.PTEDUCAT.std()
t1['apoe4_pct'] = 100 * df.APOE4_Carrier.mean()
for c in COGS:
    t1[f'{c}_mean'], t1[f'{c}_sd'] = df[c].mean(), df[c].std()
    t1[f'{c}_N'] = int(df[c].notna().sum())
t1['manual_sum_mean'], t1['manual_sum_sd'] = df.Manual_Atrophy_Sum.mean(), df.Manual_Atrophy_Sum.std()
t1['e_sum_mean'], t1['e_sum_sd'] = e42.e_sum.mean(), e42.e_sum.std()
t1['csf_N'] = int(df.dropna(subset=['ABETA', 'PTAU']).shape[0])
out['table1'] = t1


def boot_ci(y1, y2, fn, ncls=None, nb=NB, seed=42):
    np.random.seed(seed)
    y1 = np.asarray(y1); y2 = np.asarray(y2); m = len(y1)
    vals = []
    for _ in range(nb):
        i = np.random.choice(m, m, replace=True)
        vals.append(fn(y1[i], y2[i], ncls) if ncls else fn(y1[i], y2[i]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ---------------------------------------------------------------- Table 2 ---
t2 = {}
for region, ncls in LOBES:
    yv = df[region].values; pv = e42[region].values
    k = cohen_kappa_quadratic(yv, pv, ncls)
    lo, hi = boot_ci(yv, pv, cohen_kappa_quadratic, ncls)
    t2[region] = dict(kappa=k, lo=lo, hi=hi,
                      exact=100 * np.mean(yv == pv),
                      within1=100 * np.mean(abs(yv - pv) <= 1))
icc = intraclass_correlation(df.Manual_Atrophy_Sum.values, e42.e_sum.values)
lo, hi = boot_ci(df.Manual_Atrophy_Sum.values, e42.e_sum.values, intraclass_correlation)
d = e42.e_sum.values - df.Manual_Atrophy_Sum.values
t2['Total'] = dict(icc=icc, lo=lo, hi=hi, bias=float(d.mean()),
                   loa_lo=float(d.mean() - 1.96 * d.std()),
                   loa_hi=float(d.mean() + 1.96 * d.std()),
                   within2_pct=100 * np.mean(abs(d) <= 2), ge5_n=int((abs(d) >= 5).sum()))
out['table2'] = t2


# --------------------------------------------- FreeSurfer concordance --------
def rho_ci(a, b, nb=NB, seed=42):
    np.random.seed(seed); a = np.asarray(a); b = np.asarray(b); m = len(a)
    vals = [spearmanr(a[i], b[i])[0]
            for i in (np.random.choice(m, m, replace=True) for _ in range(nb))]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


ok = df.FS_Hippo_Total.notna().values
negFS = -df.loc[ok, 'FS_Hippo_Total'].values
fs = {}
for name, v in [('manual_total', df.loc[ok, 'Manual_Atrophy_Sum'].values),
                ('e_total', e42.loc[ok, 'e_sum'].values),
                ('manual_mta', (df.loc[ok, 'Hippo_Lt'] + df.loc[ok, 'Hippo_Rt']).values),
                ('e_mta', e42.loc[ok, 'e_mta'].values)]:
    r, p = spearmanr(v, negFS); lo, hi = rho_ci(v, negFS)
    fs[name] = dict(rho=r, p=p, lo=lo, hi=hi)
# Raw, uncalibrated medial temporal features vs FreeSurfer: the check that the
# window measures the anatomy the MTA scale describes.
for nm, col in [('raw_lt', 'Auto_Lt_Ratio'), ('raw_rt', 'Auto_Rt_Ratio')]:
    r, p = spearmanr(df.loc[ok, col].values, negFS)
    fs[nm] = dict(rho=r, p=p)
out['fs'] = fs

# ---------------------------------------------------------------- Table 3 ---
t3 = {}
for oc in COGS:
    sub = df[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier', oc,
              'Manual_Atrophy_Sum', 'FS_Hippo_Total']].copy()
    sub['e_sum'] = e42.e_sum.values
    sub = sub.dropna()
    cov = sub[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier']].values
    yv = sub[oc].values
    m1 = fit_ols(cov, yv)
    row = {'N': len(sub), 'cov_adjR2': m1['adj_r_squared']}
    ps = []
    for label, col, flip in [('manual', 'Manual_Atrophy_Sum', 1),
                             ('ecvrs', 'e_sum', 1),
                             ('fs', 'FS_Hippo_Total', -1)]:
        zp = (sub[col] - sub[col].mean()) / sub[col].std() * flip
        m2 = fit_ols(np.hstack([cov, zp.values.reshape(-1, 1)]), yv)
        F, p = compare_models(m1, m2)
        row[label] = dict(adjR2=m2['adj_r_squared'], F=F, p=p,
                          beta=m2['coef'][-1], se=m2['se'][-1])
        ps.append(p)
    for lab, ap in zip(['manual', 'ecvrs', 'fs'], fdr_correction(ps)):
        row[lab]['p_bh'] = float(ap)
    t3[oc] = row
out['table3'] = t3

# ---------------------------------------------------------------- Table 4 ---
csf = df.dropna(subset=['ABETA', 'PTAU'])
ii = csf.index
t4 = {'N': len(csf)}
for name, v in [('manual', csf.Manual_Atrophy_Sum.values),
                ('ecvrs', e42.loc[ii, 'e_sum'].values),
                ('fs', csf.FS_Hippo_Total.values)]:
    for bm_, bv in [('abeta', csf.ABETA.values), ('ptau', csf.PTAU.values)]:
        r, p = spearmanr(v, bv); lo, hi = rho_ci(v, bv)
        t4[f'{name}_{bm_}'] = dict(rho=r, p=p, lo=lo, hi=hi)
out['table4'] = t4

# ------------------------------------------- seed sensitivity & Table 5 ------
w = res_all[res_all.candidate == WINNER]
out['seeds'] = dict(icc_mean=w.icc_total.mean(), icc_sd=w.icc_total.std(),
                    icc_min=w.icc_total.min(), icc_max=w.icc_total.max(),
                    kappa_mean=w.mean_kappa.mean(),
                    fs_mean=w.fs_rho_total.mean(), fs_sd=w.fs_rho_total.std(),
                    per_seed=w[['seed', 'icc_total', 'mean_kappa', 'fs_rho_total',
                                'cog_mean_F', 'csf_mean_abs']].to_dict('records'))

ch = pd.read_csv(OUT_DIR / 'benchmark_choices.csv')
chw = ch[ch.candidate == WINNER]
out['lambda_selection'] = {str(k): v for k, v in
                           chw.groupby(['region', 'lambda']).size().unstack(fill_value=0).to_dict().items()}

bench = res_all.groupby('candidate')[['icc_total', 'mean_kappa', 'fs_rho_total',
                                      'cog_mean_F', 'csf_mean_abs']].mean()
bench['composite'] = pd.read_csv(OUT_DIR / 'benchmark_ranks.csv').groupby('candidate')['composite'].mean()
out['benchmark'] = bench.round(4).to_dict('index')

json.dump(out, open(OUT_DIR / 'final_results.json', 'w'), indent=1, default=float)

print(f'seed-42 headline (winner = {WINNER}):')
print(f"  ICC {t2['Total']['icc']:.4f} ({t2['Total']['lo']:.4f}-{t2['Total']['hi']:.4f})  "
      f"bias {t2['Total']['bias']:.2f}  "
      f"LoA {t2['Total']['loa_lo']:.2f}..{t2['Total']['loa_hi']:.2f}  "
      f"|d|<=2 {t2['Total']['within2_pct']:.1f}%  >=5 n={t2['Total']['ge5_n']}")
for r_, v in t2.items():
    if r_ != 'Total':
        print(f"  {r_:10s} k={v['kappa']:.3f} ({v['lo']:.3f}-{v['hi']:.3f}) "
              f"exact {v['exact']:.1f}% +/-1 {v['within1']:.1f}%")
print('  FS:', {k: round(v['rho'], 3) for k, v in fs.items()})
print('  cog e-CVRS F:', {oc: round(t3[oc]['ecvrs']['F'], 2) for oc in COGS})
print('  cog e-CVRS p_bh:', {oc: round(t3[oc]['ecvrs']['p_bh'], 4) for oc in COGS})
print('  CSF:', {k: round(v['rho'], 3) for k, v in t4.items() if k != 'N'})
print('  seeds ICC %.4f+/-%.4f  FS %.3f+/-%.3f' %
      (out['seeds']['icc_mean'], out['seeds']['icc_sd'],
       out['seeds']['fs_mean'], out['seeds']['fs_sd']))
print('saved', OUT_DIR / 'final_results.json')
