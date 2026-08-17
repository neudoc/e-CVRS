"""Candidate-algorithm benchmark for e-CVRS (7 transparent candidates x 8 seeds)."""
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

import sys, itertools, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecvrs.stats.agreement import cohen_kappa_quadratic, intraclass_correlation
from ecvrs.calibration.ordinal import train_kappa_optimized_mapping, predict_kappa_optimized_mapping
from ecvrs.stats.regression import fit_ols, compare_models

df = pd.read_csv(str(OUT_DIR / 'cohort_merged.csv'))
n = len(df)
assert n == 339

RATIO_COLS = ['Auto_Lt_Ratio','Auto_Rt_Ratio','Auto_Front_Ratio','Auto_Parietal_Ratio','Auto_Temporal_Ratio']
VOL_COLS   = ['Vol_Hippo_Lt','Vol_Hippo_Rt','Vol_Front','Vol_Parietal','Vol_Temporal']
tbv = df['Vol_TBV'].values.astype(float)
X_full = np.column_stack([df[c].values.astype(float) for c in RATIO_COLS] +
                         [df[v].values.astype(float)/tbv for v in VOL_COLS] +
                         [df['vent_norm'].values.astype(float)])   # (339, 11)

LOBES = [('Hippo_Lt',5,0),('Hippo_Rt',5,1),('Front',4,2),('Parietal',4,3),('Temporal',4,4)]
LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

def zstats(X): 
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd==0,1.0,sd); return mu, sd

def ridge_fit(Xz, y, lam):
    p = Xz.shape[1]
    A = Xz.T @ Xz + lam*np.eye(p)
    b = np.linalg.solve(A, Xz.T @ (y - y.mean()))
    return y.mean(), b

def make_design(X, variant):
    return X if variant=='linear' else np.column_stack([X, X**2])

def inner_select(Xtr, ytr, variants, lambdas, seed):
    """4-fold inner CV on training data; pick (variant, lambda) minimizing MSE."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ytr)); rng.shuffle(idx)
    folds = np.array_split(idx, 4)
    best = (None, None, np.inf)
    for v in variants:
        Dv = make_design(Xtr, v)
        for lam in lambdas:
            errs = []
            for k in range(4):
                te = folds[k]; tr = np.hstack([folds[j] for j in range(4) if j!=k])
                mu, sd = zstats(Dv[tr]); Ztr = (Dv[tr]-mu)/sd; Zte = (Dv[te]-mu)/sd
                b0, b = ridge_fit(Ztr, ytr[tr], lam)
                errs.append(np.mean((ytr[te] - (b0 + Zte@b))**2))
            m = np.mean(errs)
            if m < best[2]: best = (v, lam, m)
    return best[0], best[1]

def fit_predict(candidate, Xtr, ytr, Xte, ncls, seed):
    """Return OOF integer grades for the test fold."""
    if candidate == 'full_ols':
        v, lam = 'linear', 0.0
    elif candidate == 'full_ridge':
        v = 'linear'; _, lam = inner_select(Xtr, ytr, ['linear'], LAMBDAS, seed)
        v, lam = 'linear', lam
    elif candidate == 'full_quadratic_ridge':
        _, lam = inner_select(Xtr, ytr, ['quadratic'], LAMBDAS, seed); v = 'quadratic'
    elif candidate == 'adaptive_ridge_nested':
        v, lam = inner_select(Xtr, ytr, ['linear','quadratic'], LAMBDAS, seed)
    elif candidate == 'rank_ols':
        v, lam = 'linear', 0.0
    else:
        raise ValueError(candidate)
    Dtr = make_design(Xtr, v); Dte = make_design(Xte, v)
    mu, sd = zstats(Dtr); Ztr = (Dtr-mu)/sd; Zte = (Dte-mu)/sd
    b0, b = ridge_fit(Ztr, ytr, lam)
    str_ = b0 + Ztr@b; ste = b0 + Zte@b
    th = train_kappa_optimized_mapping(str_, ytr, ncls)
    return predict_kappa_optimized_mapping(ste, th, ncls), (v, lam)

def rank_transform_train_test(Xtr, Xte):
    """train-ECDF transform of each column (robust monotone rescale)."""
    out_tr = np.zeros_like(Xtr); out_te = np.zeros_like(Xte)
    for j in range(Xtr.shape[1]):
        xs = np.sort(Xtr[:,j])
        out_tr[:,j] = np.searchsorted(xs, Xtr[:,j], side='right')/len(xs)
        out_te[:,j] = np.searchsorted(xs, Xte[:,j], side='right')/len(xs)
    return out_tr, out_te

def run_candidate_seed(candidate, seed):
    rng = np.random.RandomState(seed)   # replicate cli.py: np.random.seed then shuffle
    np.random.seed(seed)
    idx = np.arange(n); np.random.shuffle(idx)
    folds = np.array_split(idx, 5)
    preds = {r: np.zeros(n, dtype=int) for r,_,_ in LOBES}
    choices = []
    for region, ncls, _ in LOBES:
        y = df[region].values.astype(float)
        for k in range(5):
            te = folds[k]; tr = np.hstack([folds[j] for j in range(5) if j!=k])
            Xtr, Xte = X_full[tr], X_full[te]
            if candidate == 'rank_ols':
                Xtr, Xte = rank_transform_train_test(Xtr, Xte)
            p, ch = fit_predict(candidate, Xtr, y[tr].astype(float), Xte, ncls, seed)
            preds[region][te] = p
            choices.append({'candidate':candidate,'seed':seed,'region':region,'fold':k,'variant':ch[0],'lambda':ch[1]})
    e = pd.DataFrame(preds)
    e['e_sum'] = e.sum(axis=1)
    e['e_mta'] = e['Hippo_Lt'] + e['Hippo_Rt']
    return e, choices

def evaluate(e):
    res = {}
    # fidelity
    res['icc_total'] = intraclass_correlation(df['Manual_Atrophy_Sum'].values, e['e_sum'].values)
    kaps = [cohen_kappa_quadratic(df[r].values, e[r].values, ncls) for r,ncls,_ in LOBES]
    res['mean_kappa'] = float(np.mean(kaps))
    for (r,ncls,_),k in zip(LOBES,kaps): res[f'kappa_{r}'] = k
    # FreeSurfer
    ok = df['FS_Hippo_Total'].notna().values
    res['fs_rho_total'] = spearmanr(e.loc[ok,'e_sum'], -df.loc[ok,'FS_Hippo_Total'])[0]
    res['fs_rho_mta']   = spearmanr(e.loc[ok,'e_mta'], -df.loc[ok,'FS_Hippo_Total'])[0]
    # cognition: incremental F over covariates
    cog_outcomes = ['MMSE','ADAS13','CDRSB','ADNI_MEM','ADNI_EF','ADNI_LAN']
    Fs = {}
    for oc in cog_outcomes:
        sub = df[['AGE','Gender_Dummy','PTEDUCAT','APOE4_Carrier',oc]].copy()
        sub['pred'] = e['e_sum'].values
        sub = sub.dropna()
        cov = sub[['AGE','Gender_Dummy','PTEDUCAT','APOE4_Carrier']].values
        yoc = sub[oc].values
        zp = (sub['pred']-sub['pred'].mean())/sub['pred'].std()
        m1 = fit_ols(cov, yoc)
        m2 = fit_ols(np.hstack([cov, zp.values.reshape(-1,1)]), yoc)
        F, p = compare_models(m1, m2)
        Fs[oc] = F; res[f'F_{oc}'] = F; res[f'p_{oc}'] = p
    res['cog_mean_F'] = float(np.mean(list(Fs.values())))
    # CSF
    csf = df.dropna(subset=['ABETA','PTAU'])
    ii = csf.index
    rab = spearmanr(e.loc[ii,'e_sum'], csf['ABETA'])[0]
    rpt = spearmanr(e.loc[ii,'e_sum'], csf['PTAU'])[0]
    res['csf_rho_abeta'] = rab; res['csf_rho_ptau'] = rpt
    res['csf_mean_abs'] = float((abs(rab)+abs(rpt))/2)
    return res

CANDIDATES = ['full_ols','full_ridge','full_quadratic_ridge','adaptive_ridge_nested','rank_ols']
SEEDS = [0,1,2,3,4,42,100,2024]

rows, all_choices, oof_store = [], [], {}
for cand in CANDIDATES:
    for seed in SEEDS:
        e, ch = run_candidate_seed(cand, seed)
        all_choices += ch
        r = evaluate(e); r['candidate']=cand; r['seed']=seed
        rows.append(r)
        if seed == 42: oof_store[cand] = e
        print(f"{cand:22s} seed={seed:5d} ICC={r['icc_total']:.4f} FSrho={r['fs_rho_total']:.3f} cogF={r['cog_mean_F']:.2f} csf={r['csf_mean_abs']:.3f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(str(OUT_DIR / 'benchmark_results.csv'), index=False)
pd.DataFrame(all_choices).to_csv(str(OUT_DIR / 'benchmark_choices.csv'), index=False)
for cand,e in oof_store.items(): e.to_csv(fstr(OUT_DIR / 'oof_seed42_{cand}.csv'), index=False)

# ---- composite ranking (equal weight across 3 external domains, per seed) ----
doms = ['fs_rho_total','cog_mean_F','csf_mean_abs']
ranks = []
for seed in SEEDS:
    s = res[res.seed==seed].set_index('candidate')
    rk = pd.DataFrame({d: s[d].rank(ascending=False) for d in doms})
    rk['composite'] = rk.mean(axis=1); rk['seed']=seed
    ranks.append(rk.reset_index())
rk = pd.concat(ranks)
summary = rk.groupby('candidate')['composite'].mean().sort_values()
print('\n=== COMPOSITE MEAN RANK (lower=better) ===')
print(summary.to_string())
rk.to_csv(str(OUT_DIR / 'benchmark_ranks.csv'), index=False)

print('\n=== per-domain means over 8 seeds ===')
print(res.groupby('candidate')[['icc_total','mean_kappa','fs_rho_total','fs_rho_mta','cog_mean_F','csf_mean_abs','csf_rho_abeta','csf_rho_ptau']].mean().round(4).to_string())
