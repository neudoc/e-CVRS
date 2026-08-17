"""Additional candidate models for the e-CVRS calibration benchmark."""
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
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
exec(open(Path(__file__).parent / "benchmark.py").read().split("CANDIDATES = ")[0])
SEEDS = [0,1,2,3,4,42,100,2024]

# extra candidates
def fit_predict2(candidate, Xtr, ytr, Xte, ncls, seed):
    if candidate == 'full_quadratic_ols':
        v, lam = 'quadratic', 0.0
    elif candidate == 'adaptive_ols_nested':
        v, lam = inner_select(Xtr, ytr, ['linear','quadratic'], [0.0], seed)
    elif candidate == 'asym_ols':
        v, lam = 'linear', 0.0
    else:
        return fit_predict(candidate, Xtr, ytr, Xte, ncls, seed)
    Dtr = make_design(Xtr, v); Dte = make_design(Xte, v)
    mu, sd = zstats(Dtr); Ztr=(Dtr-mu)/sd; Zte=(Dte-mu)/sd
    b0,b = ridge_fit(Ztr, ytr, lam)
    th = train_kappa_optimized_mapping(b0+Ztr@b, ytr, ncls)
    return predict_kappa_optimized_mapping(b0+Zte@b, th, ncls), (v,lam)

def add_asym(X):
    # asymmetry features: ratio and volume L-R differences
    return np.column_stack([X, X[:,0]-X[:,1], X[:,5]-X[:,6]])

def run2(candidate, seed):
    np.random.seed(seed)
    idx = np.arange(n); np.random.shuffle(idx)
    folds = np.array_split(idx, 5)
    preds = {r: np.zeros(n,dtype=int) for r,_,_ in LOBES}
    for region, ncls, _ in LOBES:
        y = df[region].values.astype(float)
        for k in range(5):
            te = folds[k]; tr = np.hstack([folds[j] for j in range(5) if j!=k])
            Xtr, Xte = X_full[tr], X_full[te]
            if candidate=='asym_ols': Xtr, Xte = add_asym(Xtr), add_asym(Xte)
            p,_ = fit_predict2(candidate, Xtr, y[tr], Xte, ncls, seed)
            preds[region][te] = p
    e = pd.DataFrame(preds); e['e_sum']=e.sum(axis=1); e['e_mta']=e['Hippo_Lt']+e['Hippo_Rt']
    return e

rows=[]; oof={}
for cand in ['full_quadratic_ols','adaptive_ols_nested']:
    for seed in SEEDS:
        e = run2(cand, seed)
        r = evaluate(e); r['candidate']=cand; r['seed']=seed; rows.append(r)
        if seed==42: oof[cand]=e
        print(f"{cand:22s} seed={seed:5d} ICC={r['icc_total']:.4f} FSrho={r['fs_rho_total']:.3f} cogF={r['cog_mean_F']:.2f} csf={r['csf_mean_abs']:.3f}", flush=True)
extra = pd.DataFrame(rows)
extra.to_csv(str(OUT_DIR / 'benchmark_results_extra.csv'), index=False)
for c,e in oof.items(): e.to_csv(fstr(OUT_DIR / 'oof_seed42_{c}.csv'), index=False)

allres = pd.concat([pd.read_csv(str(OUT_DIR / 'benchmark_results.csv')), extra], ignore_index=True)
allres.to_csv(str(OUT_DIR / 'benchmark_results_all.csv'), index=False)
doms=['fs_rho_total','cog_mean_F','csf_mean_abs']
ranks=[]
for seed in SEEDS:
    s = allres[allres.seed==seed].set_index('candidate')
    rk = pd.DataFrame({d: s[d].rank(ascending=False) for d in doms})
    rk['composite']=rk.mean(axis=1); rk['seed']=seed; ranks.append(rk.reset_index())
rk=pd.concat(ranks); rk.to_csv(str(OUT_DIR / 'benchmark_ranks_all.csv'), index=False)
print('\n=== COMPOSITE MEAN RANK, 8 candidates ===')
print(rk.groupby('candidate')['composite'].mean().sort_values().round(3).to_string())
print('\n=== domain means ===')
print(allres.groupby('candidate')[['icc_total','fs_rho_total','cog_mean_F','csf_mean_abs']].mean().round(4).to_string())
# head-to-head per seed between top candidates
print('\n=== per-seed composite: adaptive_ridge_nested vs full_ols vs full_quadratic_ols ===')
piv = rk.pivot(index='seed', columns='candidate', values='composite')
print(piv.round(2).to_string())
