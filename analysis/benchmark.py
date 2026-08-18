"""Candidate-algorithm benchmark for e-CVRS (5 transparent candidates x 8 seeds).

The calibration algorithm is selected by a PRESPECIFIED equal-weight composite
of three external validity domains, each computed strictly on held-out
predictions:

  1. Spearman rho between the automated total score and FreeSurfer v7
     hippocampal volume (structural),
  2. mean covariate-adjusted incremental F across six cognitive outcomes
     (clinical),
  3. mean absolute Spearman rho with CSF Abeta42 and p-tau181 (biological).

Candidates are ranked within each fold-assignment seed on each domain; the
equal-weight mean rank, averaged over seeds, picks the winner. Agreement with
the expert ratings is reported but deliberately NOT used for selection, so the
instrument is not tuned to a single imperfect reference standard.

Outputs (in ECVRS_OUT_DIR):
  benchmark_results.csv   per-candidate, per-seed metrics
  benchmark_choices.csv   design/lambda chosen in every outer training fold
  benchmark_ranks.csv     per-seed domain ranks and composite
  oof_seed42_<cand>.csv   out-of-fold grades at the primary seed
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Override with environment variables so the scripts run from a clone
# without editing code:
#   ECVRS_DATA_DIR  directory holding the ADNI-derived inputs
#   ECVRS_OUT_DIR   directory for intermediate and final analysis outputs
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecvrs.stats.agreement import cohen_kappa_quadratic, intraclass_correlation
from ecvrs.calibration.ordinal import train_kappa_optimized_mapping, predict_kappa_optimized_mapping
from ecvrs.stats.regression import fit_ols, compare_models

RATIO_COLS = ['Auto_Lt_Ratio', 'Auto_Rt_Ratio', 'Auto_Front_Ratio',
              'Auto_Parietal_Ratio', 'Auto_Temporal_Ratio']
VOL_COLS = ['Vol_Hippo_Lt', 'Vol_Hippo_Rt', 'Vol_Front', 'Vol_Parietal', 'Vol_Temporal']
LOBES = [('Hippo_Lt', 5), ('Hippo_Rt', 5), ('Front', 4), ('Parietal', 4), ('Temporal', 4)]
COGS = ['MMSE', 'ADAS13', 'CDRSB', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN']
LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2, 3, 4, 42, 100, 2024]
CANDIDATES = ['full_ols', 'full_ridge', 'full_quadratic_ridge',
              'adaptive_ridge_nested', 'rank_ols']
DOMAINS = ['fs_rho_total', 'cog_mean_F', 'csf_mean_abs']


def build_X(df):
    """The 11-feature panel: 5 CSF ratios, 5 TBV-normalized volumes, ventricular fraction."""
    tbv = df['Vol_TBV'].values.astype(float)
    return np.column_stack([df[c].values.astype(float) for c in RATIO_COLS] +
                           [df[v].values.astype(float) / tbv for v in VOL_COLS] +
                           [df['vent_norm'].values.astype(float)])


def zstats(X):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    return mu, sd


def ridge_fit(Xz, y, lam):
    p = Xz.shape[1]
    return y.mean(), np.linalg.solve(Xz.T @ Xz + lam * np.eye(p), Xz.T @ (y - y.mean()))


def make_design(X, variant):
    return X if variant == 'linear' else np.hstack([X, X ** 2])


def inner_select(Xtr, ytr, variants, lambdas, seed):
    """4-fold inner CV, run strictly inside the outer training fold."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ytr)); rng.shuffle(idx)
    folds = np.array_split(idx, 4)
    best = (None, None, np.inf)
    for v in variants:
        D = make_design(Xtr, v)
        for lam in lambdas:
            errs = []
            for k in range(4):
                te = folds[k]; tr = np.hstack([folds[j] for j in range(4) if j != k])
                mu, sd = zstats(D[tr]); Ztr = (D[tr] - mu) / sd; Zte = (D[te] - mu) / sd
                try:
                    b0, b = ridge_fit(Ztr, ytr[tr], lam)
                except np.linalg.LinAlgError:
                    errs.append(np.inf); continue
                errs.append(np.mean((ytr[te] - (b0 + Zte @ b)) ** 2))
            e = float(np.mean(errs))
            if e < best[2]:
                best = (v, lam, e)
    return best[0], best[1]


def fit_predict(candidate, Xtr, ytr, Xte, ncls, seed):
    if candidate == 'adaptive_ridge_nested':
        v, lam = inner_select(Xtr, ytr, ['linear', 'quadratic'], LAMBDAS, seed)
    elif candidate == 'full_ols':
        v, lam = 'linear', 0.0
    elif candidate == 'full_ridge':
        v, lam = 'linear', None
    elif candidate == 'full_quadratic_ridge':
        v, lam = 'quadratic', None
    elif candidate == 'rank_ols':
        v, lam = 'linear', 0.0
    else:
        raise ValueError(candidate)
    if lam is None:
        _, lam = inner_select(Xtr, ytr, [v], LAMBDAS, seed)

    Dtr = make_design(Xtr, v); Dte = make_design(Xte, v)
    mu, sd = zstats(Dtr); Ztr = (Dtr - mu) / sd; Zte = (Dte - mu) / sd
    try:
        b0, b = ridge_fit(Ztr, ytr, lam)
    except np.linalg.LinAlgError:
        b0, b = ridge_fit(Ztr, ytr, 0.01)
    thr = train_kappa_optimized_mapping(b0 + Ztr @ b, ytr, ncls)
    return predict_kappa_optimized_mapping(b0 + Zte @ b, thr, ncls), (v, lam)


def rank_transform_train_test(Xtr, Xte):
    out_tr = np.zeros_like(Xtr); out_te = np.zeros_like(Xte)
    for j in range(Xtr.shape[1]):
        xs = np.sort(Xtr[:, j])
        out_tr[:, j] = np.searchsorted(xs, Xtr[:, j], side='right') / len(xs)
        out_te[:, j] = np.searchsorted(xs, Xte[:, j], side='right') / len(xs)
    return out_tr, out_te


def run_candidate_seed(df, X_full, candidate, seed):
    """5-fold outer CV; every fit happens inside the training fold."""
    n = len(df)
    np.random.seed(seed)
    idx = np.arange(n); np.random.shuffle(idx)
    folds = np.array_split(idx, 5)
    preds = {r: np.zeros(n, dtype=int) for r, _ in LOBES}
    choices = []
    for region, ncls in LOBES:
        y = df[region].values.astype(float)
        for k in range(5):
            te = folds[k]; tr = np.hstack([folds[j] for j in range(5) if j != k])
            Xtr, Xte = X_full[tr], X_full[te]
            if candidate == 'rank_ols':
                Xtr, Xte = rank_transform_train_test(Xtr, Xte)
            p, ch = fit_predict(candidate, Xtr, y[tr], Xte, ncls, seed)
            preds[region][te] = p
            choices.append({'candidate': candidate, 'seed': seed, 'region': region,
                            'fold': k, 'variant': ch[0], 'lambda': ch[1]})
    e = pd.DataFrame(preds)
    e['e_sum'] = e.sum(axis=1)
    e['e_mta'] = e['Hippo_Lt'] + e['Hippo_Rt']
    return e, choices


def evaluate(df, e):
    res = {}
    res['icc_total'] = intraclass_correlation(df['Manual_Atrophy_Sum'].values, e['e_sum'].values)
    kaps = [cohen_kappa_quadratic(df[r].values, e[r].values, ncls) for r, ncls in LOBES]
    res['mean_kappa'] = float(np.mean(kaps))
    for (r, _), k in zip(LOBES, kaps):
        res[f'kappa_{r}'] = k

    ok = df['FS_Hippo_Total'].notna().values
    res['fs_rho_total'] = spearmanr(e.loc[ok, 'e_sum'], -df.loc[ok, 'FS_Hippo_Total'])[0]
    res['fs_rho_mta'] = spearmanr(e.loc[ok, 'e_mta'], -df.loc[ok, 'FS_Hippo_Total'])[0]

    Fs = {}
    for oc in COGS:
        sub = df[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier', oc]].copy()
        sub['pred'] = e['e_sum'].values
        sub = sub.dropna()
        cov = sub[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier']].values
        yoc = sub[oc].values
        zp = (sub['pred'] - sub['pred'].mean()) / sub['pred'].std()
        F, p = compare_models(fit_ols(cov, yoc),
                              fit_ols(np.hstack([cov, zp.values.reshape(-1, 1)]), yoc))
        Fs[oc] = F; res[f'F_{oc}'] = F; res[f'p_{oc}'] = p
    res['cog_mean_F'] = float(np.mean(list(Fs.values())))

    csf = df.dropna(subset=['ABETA', 'PTAU'])
    ii = csf.index
    rab = spearmanr(e.loc[ii, 'e_sum'], csf['ABETA'])[0]
    rpt = spearmanr(e.loc[ii, 'e_sum'], csf['PTAU'])[0]
    res['csf_rho_abeta'] = rab; res['csf_rho_ptau'] = rpt
    res['csf_mean_abs'] = float((abs(rab) + abs(rpt)) / 2)
    return res


def main():
    df = pd.read_csv(OUT_DIR / 'cohort_merged.csv')
    print('cohort N =', len(df))
    X_full = build_X(df)

    rows, all_choices, oof_store = [], [], {}
    for cand in CANDIDATES:
        for seed in SEEDS:
            e, ch = run_candidate_seed(df, X_full, cand, seed)
            all_choices += ch
            r = evaluate(df, e); r['candidate'] = cand; r['seed'] = seed
            rows.append(r)
            if seed == 42:
                oof_store[cand] = e
        m = pd.DataFrame([x for x in rows if x['candidate'] == cand]).mean(numeric_only=True)
        print(f"{cand:22s} ICC={m['icc_total']:.4f} FS={m['fs_rho_total']:.3f} "
              f"cogF={m['cog_mean_F']:.2f} csf={m['csf_mean_abs']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'benchmark_results.csv', index=False)
    pd.DataFrame(all_choices).to_csv(OUT_DIR / 'benchmark_choices.csv', index=False)
    for cand, e in oof_store.items():
        e.to_csv(OUT_DIR / f'oof_seed42_{cand}.csv', index=False)

    ranks = []
    for seed in SEEDS:
        s = res[res.seed == seed].set_index('candidate')
        rk = pd.DataFrame({d: s[d].rank(ascending=False) for d in DOMAINS})
        rk['composite'] = rk.mean(axis=1); rk['seed'] = seed
        ranks.append(rk.reset_index())
    rk = pd.concat(ranks)
    rk.to_csv(OUT_DIR / 'benchmark_ranks.csv', index=False)

    summary = rk.groupby('candidate')['composite'].mean().sort_values()
    print('\n=== COMPOSITE MEAN RANK (lower = better) ===')
    print(summary.to_string())
    print('\nWINNER:', summary.index[0], '(published: full_quadratic_ridge, composite 1.67)')


if __name__ == '__main__':
    main()
