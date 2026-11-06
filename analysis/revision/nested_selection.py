"""Selection-honest (nested) evaluation of the calibration-candidate choice.

Reviewer 2, major comment 1: the calibration candidate was chosen by a composite of
FreeSurfer / cognition / CSF, and the same domains were then reported as validity.

Here the WHOLE selection step is moved inside each outer training fold:
  outer 5-fold CV (same fold assignments as the published benchmark, 8 seeds)
    for each outer training fold (~271 subjects):
      inner 5-fold CV -> inner out-of-fold grades for every candidate
      compute the three prespecified domains on the TRAINING subjects only
      equal-weight mean rank -> pick a candidate
      refit the picked candidate on the full outer training fold
      predict the held-out outer fold
The held-out outer-fold subjects never influence which candidate is chosen, so the
resulting out-of-fold metrics include the selection step (no winner's-curse optimism).
Also reported: the selection frequency of each candidate and the fixed-winner
(published) metrics on the same seeds, so the optimism is quantified directly.
"""
import sys, json, time
REPO = sys.argv[1]
sys.path.insert(0, REPO + '/analysis'); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import benchmark as B
from ecvrs.stats.regression import fit_ols, compare_models

OUT = B.OUT_DIR
df = pd.read_csv(OUT / 'cohort_merged.csv'); X_full = B.build_X(df)
N_INNER = 5


def domains(sub, esum):
    """The three prespecified selection domains, computed on a subset of subjects."""
    ok = sub['FS_Hippo_Total'].notna().values
    fs = spearmanr(esum[ok], -sub.loc[ok, 'FS_Hippo_Total'].values)[0]
    Fs = []
    for oc in B.COGS:
        t = sub[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier', oc]].copy()
        t['pred'] = esum
        t = t.dropna()
        cov = t[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier']].values
        zp = ((t['pred'] - t['pred'].mean()) / t['pred'].std()).values.reshape(-1, 1)
        F, _ = compare_models(fit_ols(cov, t[oc].values), fit_ols(np.hstack([cov, zp]), t[oc].values))
        Fs.append(F)
    c = sub[['ABETA', 'PTAU']].notna().all(axis=1).values
    ra = spearmanr(esum[c], sub.loc[c, 'ABETA'].values)[0]
    rp = spearmanr(esum[c], sub.loc[c, 'PTAU'].values)[0]
    return {'fs_rho_total': fs, 'cog_mean_F': float(np.mean(Fs)), 'csf_mean_abs': (abs(ra) + abs(rp)) / 2}


def fit_pred_region(cand, Xtr, ytr, Xte, ncls, seed):
    if cand == 'rank_ols':
        Xtr, Xte = B.rank_transform_train_test(Xtr, Xte)
    p, ch = B.fit_predict(cand, Xtr, ytr, Xte, ncls, seed)
    return p


def select_inside(tr_idx, seed, k_outer):
    """Run the prespecified composite selection using ONLY the outer training subjects."""
    sub = df.iloc[tr_idx].reset_index(drop=True)
    Xs = X_full[tr_idx]
    rng = np.random.RandomState(seed * 100 + k_outer)
    idx = np.arange(len(sub)); rng.shuffle(idx)
    ifolds = np.array_split(idx, N_INNER)
    scores = {}
    for cand in B.CANDIDATES:
        esum = np.zeros(len(sub))
        for region, ncls in B.LOBES:
            y = sub[region].values.astype(float)
            for j in range(N_INNER):
                te = ifolds[j]; tr = np.hstack([ifolds[q] for q in range(N_INNER) if q != j])
                esum[te] += fit_pred_region(cand, Xs[tr], y[tr], Xs[te], ncls, seed)
        scores[cand] = domains(sub, esum)
    sc = pd.DataFrame(scores).T
    rk = pd.DataFrame({d: sc[d].rank(ascending=False) for d in B.DOMAINS})
    rk['composite'] = rk.mean(axis=1)
    winner = rk['composite'].idxmin()
    return winner, rk['composite'].to_dict()


def run_seed(seed):
    n = len(df)
    np.random.seed(seed)                      # identical outer folds to the benchmark
    idx = np.arange(n); np.random.shuffle(idx)
    folds = np.array_split(idx, 5)
    preds = {r: np.zeros(n, dtype=int) for r, _ in B.LOBES}
    picks = []
    for k in range(5):
        te = folds[k]; tr = np.hstack([folds[j] for j in range(5) if j != k])
        win, comp = select_inside(tr, seed, k)
        picks.append({'seed': seed, 'fold': k, 'picked': win, **{f'rank_{c}': v for c, v in comp.items()}})
        for region, ncls in B.LOBES:
            y = df[region].values.astype(float)
            preds[region][te] = fit_pred_region(win, X_full[tr], y[tr], X_full[te], ncls, seed)
    e = pd.DataFrame(preds)
    e['e_sum'] = e.sum(axis=1); e['e_mta'] = e['Hippo_Lt'] + e['Hippo_Rt']
    return e, picks


if __name__ == '__main__':
    bench = pd.read_csv(OUT / 'benchmark_results.csv')
    rows, allpicks = [], []
    for s in B.SEEDS:
        t = time.time()
        e, picks = run_seed(s)
        r = B.evaluate(df, e); r['seed'] = s
        rows.append(r); allpicks += picks
        e.to_csv(OUT / f'rev_nested_oof_seed{s}.csv', index=False)
        w = bench[(bench.candidate == 'full_quadratic_ridge') & (bench.seed == s)].iloc[0]
        print(f"seed {s:5d} {time.time()-t:5.0f}s picks={[p['picked'] for p in picks]}  "
              f"nested ICC {r['icc_total']:.4f} FS {r['fs_rho_total']:.3f} F {r['cog_mean_F']:.2f} CSF {r['csf_mean_abs']:.3f} | "
              f"fixed ICC {w.icc_total:.4f} FS {w.fs_rho_total:.3f} F {w.cog_mean_F:.2f} CSF {w.csf_mean_abs:.3f}", flush=True)
    nres = pd.DataFrame(rows); pk = pd.DataFrame(allpicks)
    nres.to_csv(OUT / 'rev_nested_results.csv', index=False); pk.to_csv(OUT / 'rev_nested_picks.csv', index=False)
    fixed = bench[bench.candidate == 'full_quadratic_ridge']
    cols = ['icc_total', 'mean_kappa', 'fs_rho_total', 'fs_rho_mta', 'cog_mean_F', 'csf_mean_abs', 'csf_rho_abeta', 'csf_rho_ptau'] + \
           [f'F_{c}' for c in B.COGS] + [f'kappa_{r}' for r, _ in B.LOBES]
    summ = pd.DataFrame({'nested_mean': nres[cols].mean(), 'nested_sd': nres[cols].std(),
                         'fixed_mean': fixed[cols].mean(), 'fixed_sd': fixed[cols].std()})
    summ['optimism_fixed_minus_nested'] = summ['fixed_mean'] - summ['nested_mean']
    summ.to_csv(OUT / 'rev_nested_summary.csv')
    print('\nSelection frequency (40 outer training folds):')
    print(pk['picked'].value_counts().to_string())
    print('\n', summ.round(4).to_string())
