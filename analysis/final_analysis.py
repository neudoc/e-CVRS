"""Final results for the selected method (adaptive nested linear/quadratic ridge)."""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Override with environment variables so the scripts run from a clone
# without editing code:
#   ECVRS_DATA_DIR  directory holding the ADNI-derived inputs
#                   (e-CVRS_automated_scores.csv, ADNI_MRI_rating.xlsx,
#                    UCSF_FS.csv, UWNPSYCHSUM_*.csv, UPENNBIOMK9_*.csv)
#   ECVRS_OUT_DIR   directory for intermediate and final analysis outputs
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
exec(open(Path(__file__).parent / "benchmark.py").read().split("CANDIDATES = ")[0])
SEEDS = [0,1,2,3,4,42,100,2024]
WINNER = 'adaptive_ridge_nested'
rng_global = np.random.RandomState(7)
NB = 2000

e42 = pd.read_csv(fstr(OUT_DIR / 'oof_seed42_{WINNER}.csv'))
res_all = pd.read_csv(str(OUT_DIR / 'benchmark_results_all.csv'))
out = {}

# ---------- Table 1 ----------
t1 = {}
t1['N'] = int(n)
t1['age_mean'], t1['age_sd'] = df.AGE.mean(), df.AGE.std()
t1['age_min'], t1['age_max'] = df.AGE.min(), df.AGE.max()
t1['female_pct'] = 100*df.Gender_Dummy.mean()
t1['educ_mean'], t1['educ_sd'] = df.PTEDUCAT.mean(), df.PTEDUCAT.std()
t1['apoe4_pct'] = 100*df.APOE4_Carrier.mean()
t1['apoe4_1_pct'] = 100*(df.APOE4==1).mean(); t1['apoe4_2_pct'] = 100*(df.APOE4==2).mean()
for c in ['MMSE','ADAS13','CDRSB','ADNI_MEM','ADNI_EF','ADNI_LAN']:
    t1[f'{c}_mean'], t1[f'{c}_sd'] = df[c].mean(), df[c].std()
    t1[f'{c}_N'] = int(df[c].notna().sum())
t1['MMSE_min'], t1['MMSE_max'] = int(df.MMSE.min()), int(df.MMSE.max())
t1['manual_sum_mean'], t1['manual_sum_sd'] = df.Manual_Atrophy_Sum.mean(), df.Manual_Atrophy_Sum.std()
t1['e_sum_mean'], t1['e_sum_sd'] = e42.e_sum.mean(), e42.e_sum.std()
t1['csf_N'] = int(df.dropna(subset=['ABETA','PTAU']).shape[0])
out['table1'] = t1

# ---------- Table 2: agreement with bootstrap CIs (seed 42 OOF) ----------
def boot_ci(y1, y2, fn, ncls=None, nb=NB, seed=42):
    np.random.seed(seed)
    vals=[]
    y1=np.asarray(y1); y2=np.asarray(y2); m=len(y1)
    for _ in range(nb):
        i=np.random.choice(m,m,replace=True)
        vals.append(fn(y1[i],y2[i],ncls) if ncls else fn(y1[i],y2[i]))
    return np.percentile(vals,2.5), np.percentile(vals,97.5)

t2 = {}
for region, ncls, _ in LOBES:
    yv = df[region].values; pv = e42[region].values
    k = cohen_kappa_quadratic(yv, pv, ncls)
    lo,hi = boot_ci(yv,pv,cohen_kappa_quadratic,ncls)
    exact = 100*np.mean(yv==pv); adj = 100*np.mean(abs(yv-pv)<=1)
    t2[region] = dict(kappa=k, lo=lo, hi=hi, exact=exact, within1=adj)
icc = intraclass_correlation(df.Manual_Atrophy_Sum.values, e42.e_sum.values)
lo,hi = boot_ci(df.Manual_Atrophy_Sum.values, e42.e_sum.values, intraclass_correlation)
d = e42.e_sum.values - df.Manual_Atrophy_Sum.values
t2['Total'] = dict(icc=icc, lo=lo, hi=hi,
                   bias=float(d.mean()), loa_lo=float(d.mean()-1.96*d.std()), loa_hi=float(d.mean()+1.96*d.std()),
                   within2_pct=100*np.mean(abs(d)<=2), ge5_n=int((abs(d)>=5).sum()))
out['table2'] = t2

# ---------- FS concordance (seed 42) ----------
ok = df.FS_Hippo_Total.notna().values
def rho_ci(a, b, nb=NB, seed=42):
    np.random.seed(seed); a=np.asarray(a); b=np.asarray(b); m=len(a); vals=[]
    for _ in range(nb):
        i=np.random.choice(m,m,replace=True); vals.append(spearmanr(a[i],b[i])[0])
    return np.percentile(vals,2.5), np.percentile(vals,97.5)
fs = {}
negFS = -df.loc[ok,'FS_Hippo_Total'].values
for name, v in [('manual_total', df.loc[ok,'Manual_Atrophy_Sum'].values),
                ('e_total', e42.loc[ok,'e_sum'].values),
                ('manual_mta', (df.loc[ok,'Hippo_Lt']+df.loc[ok,'Hippo_Rt']).values),
                ('e_mta', e42.loc[ok,'e_mta'].values)]:
    r,p = spearmanr(v, negFS); lo,hi = rho_ci(v, negFS)
    fs[name] = dict(rho=r, p=p, lo=lo, hi=hi)
out['fs'] = fs

# ---------- Table 3: hierarchical regressions, BH across 3 imaging terms per outcome ----------
from ecvrs.stats.regression import fdr_correction
cogs = ['MMSE','ADAS13','CDRSB','ADNI_MEM','ADNI_EF','ADNI_LAN']
t3 = {}
for oc in cogs:
    sub = df[['AGE','Gender_Dummy','PTEDUCAT','APOE4_Carrier',oc,'Manual_Atrophy_Sum','FS_Hippo_Total']].copy()
    sub['e_sum'] = e42.e_sum.values
    sub = sub.dropna()
    cov = sub[['AGE','Gender_Dummy','PTEDUCAT','APOE4_Carrier']].values
    yv = sub[oc].values
    m1 = fit_ols(cov, yv)
    row = {'N': len(sub), 'cov_adjR2': m1['adj_r_squared']}
    ps = []
    for label, col, flip in [('manual','Manual_Atrophy_Sum',1),('ecvrs','e_sum',1),('fs','FS_Hippo_Total',-1)]:
        zp = (sub[col]-sub[col].mean())/sub[col].std()*flip
        m2 = fit_ols(np.hstack([cov, zp.values.reshape(-1,1)]), yv)
        F,p = compare_models(m1,m2)
        row[label] = dict(adjR2=m2['adj_r_squared'], F=F, p=p,
                          beta=m2['coef'][-1], se=m2['se'][-1])
        ps.append(p)
    adj = fdr_correction(ps)
    for lab, ap in zip(['manual','ecvrs','fs'], adj): row[lab]['p_bh'] = float(ap)
    t3[oc] = row
out['table3'] = t3

# ---------- Table 4: CSF ----------
csf = df.dropna(subset=['ABETA','PTAU'])
ii = csf.index
t4 = {}
for name, v in [('manual', csf.Manual_Atrophy_Sum.values),
                ('ecvrs', e42.loc[ii,'e_sum'].values),
                ('fs', csf.FS_Hippo_Total.values)]:
    for bm_, bv in [('abeta', csf.ABETA.values), ('ptau', csf.PTAU.values)]:
        r,p = spearmanr(v, bv); lo,hi = rho_ci(v, bv)
        t4[f'{name}_{bm_}'] = dict(rho=r, p=p, lo=lo, hi=hi)
t4['N'] = len(csf)
out['table4'] = t4

# ---------- seed sensitivity for winner ----------
w = res_all[res_all.candidate==WINNER]
out['seeds'] = dict(icc_mean=w.icc_total.mean(), icc_sd=w.icc_total.std(), icc_min=w.icc_total.min(), icc_max=w.icc_total.max(),
                    kappa_mean=w.mean_kappa.mean(), kappa_sd=w.mean_kappa.std(),
                    fs_mean=w.fs_rho_total.mean(), fs_sd=w.fs_rho_total.std(),
                    per_seed=w[['seed','icc_total','mean_kappa','fs_rho_total','cog_mean_F','csf_mean_abs']].to_dict('records'))

# ---------- variant/lambda selection frequencies (winner) ----------
ch = pd.read_csv(str(OUT_DIR / 'benchmark_choices.csv'))
chw = ch[ch.candidate==WINNER]
sel = chw.groupby(['region','variant']).size().unstack(fill_value=0)
out['variant_selection'] = sel.to_dict()
lam = chw.groupby(['region','lambda']).size().unstack(fill_value=0)
out['lambda_selection'] = {str(k):v for k,v in lam.to_dict().items()}

# ---------- final full-data deployment model coefficients (seed42-independent) ----------
coefs = {}
for region, ncls, _ in LOBES:
    y = df[region].values.astype(float)
    v, lam_ = inner_select(X_full, y, ['linear','quadratic'], LAMBDAS, 42)
    D = make_design(X_full, v); mu, sd = zstats(D); Z=(D-mu)/sd
    b0,b = ridge_fit(Z, y, lam_)
    coefs[region] = dict(variant=v, lam=lam_, intercept=float(b0), betas=[float(x) for x in b])
out['final_model'] = coefs

json.dump(out, open(str(OUT_DIR / 'final_results.json'),'w'), indent=1, default=float)
print('WINNER seed-42 headline:')
print(f"  ICC {icc:.3f} ({lo:.3f}-{hi:.3f})")
for r_,v in t2.items():
    if r_!='Total': print(f"  {r_:10s} kappa {v['kappa']:.3f} ({v['lo']:.3f}-{v['hi']:.3f}) exact {v['exact']:.0f}% ±1 {v['within1']:.0f}%")
print('  FS:', {k:round(v['rho'],3) for k,v in fs.items()})
print('  Table3 e-CVRS F:', {oc:round(t3[oc]['ecvrs']['F'],1) for oc in cogs})
print('  Table3 e-CVRS p_bh:', {oc:round(t3[oc]['ecvrs']['p_bh'],4) for oc in cogs})
print('  CSF:', {k:round(v['rho'],3) for k,v in t4.items() if k!='N'})
print('  seeds ICC mean', round(out['seeds']['icc_mean'],4), 'sd', round(out['seeds']['icc_sd'],4))
print('  variant sel:', out['variant_selection'])
