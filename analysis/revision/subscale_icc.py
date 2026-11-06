"""Same-definition agreement: e-CVRS vs expert on the CVRS sub-scale sums used in
Jang et al. 2018 S1 Table (hippocampal atrophy L+R 0-8; cortical atrophy F+P+T 0-9),
plus the 0-17 atrophy total. ICC(2,1) absolute agreement, bootstrap 95% CI.
Seed 42 (primary) and mean/SD over the 8 fold-assignment seeds."""
import sys, os, json, pickle
REPO = sys.argv[1]
sys.path.insert(0, REPO + '/analysis'); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import benchmark as B
from ecvrs.stats.agreement import intraclass_correlation

OUT = B.OUT_DIR
df = pd.read_csv(OUT / 'cohort_merged.csv'); X = B.build_X(df)

def defs(t):
    return {'hippocampal_sum_0_8': t['Hippo_Lt'] + t['Hippo_Rt'],
            'cortical_sum_0_9': t['Front'] + t['Parietal'] + t['Temporal'],
            'atrophy_total_0_17': t['Hippo_Lt'] + t['Hippo_Rt'] + t['Front'] + t['Parietal'] + t['Temporal']}

man = defs(df)

def boot(a, b, nb=2000, seed=42):
    rng = np.random.RandomState(seed); a = np.asarray(a, float); b = np.asarray(b, float); m = len(a)
    v = [intraclass_correlation(a[i], b[i]) for i in (rng.randint(0, m, m) for _ in range(nb))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

per_seed = []
oof42 = None
for s in B.SEEDS:
    e, _ = B.run_candidate_seed(df, X, 'full_quadratic_ridge', s)
    if s == 42: oof42 = e
    a = defs(e)
    per_seed.append({'seed': s, **{k: intraclass_correlation(man[k].values, a[k].values) for k in man}})

ps = pd.DataFrame(per_seed)
a42 = defs(oof42)
res = {'seed42': {}, 'seeds_mean': {}, 'seeds_sd': {}, 'seeds_min': {}, 'seeds_max': {}}
for k in man:
    icc = intraclass_correlation(man[k].values, a42[k].values)
    lo, hi = boot(man[k].values, a42[k].values)
    d = a42[k].values - man[k].values
    res['seed42'][k] = dict(icc=icc, lo=lo, hi=hi, bias=float(d.mean()),
                            loa_lo=float(d.mean() - 1.96 * d.std()), loa_hi=float(d.mean() + 1.96 * d.std()),
                            manual_mean=float(man[k].mean()), manual_sd=float(man[k].std()),
                            auto_mean=float(a42[k].mean()), auto_sd=float(a42[k].std()))
    res['seeds_mean'][k] = float(ps[k].mean()); res['seeds_sd'][k] = float(ps[k].std())
    res['seeds_min'][k] = float(ps[k].min()); res['seeds_max'][k] = float(ps[k].max())
res['per_seed'] = per_seed
res['expert_reference_JANG2018_S1'] = {
    'hippocampal_inter': [0.864, 0.785, 0.920], 'hippocampal_intra': [0.888, 0.803, 0.938],
    'cortical_inter': [0.904, 0.847, 0.944], 'cortical_intra': [0.942, 0.899, 0.969],
    'cvrs_total_0_30_inter': [0.941, 0.902, 0.965], 'cvrs_total_0_30_intra': [0.936, 0.889, 0.966]}
json.dump(res, open(OUT / 'rev_subscale_icc.json', 'w'), indent=1)
for k in man:
    r = res['seed42'][k]
    print(f"{k:22s} ICC {r['icc']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})  8-seed {res['seeds_mean'][k]:.3f}+/-{res['seeds_sd'][k]:.3f}"
          f"  bias {r['bias']:+.2f} LoA {r['loa_lo']:.2f}..{r['loa_hi']:.2f}  manual {r['manual_mean']:.2f}+/-{r['manual_sd']:.2f}")
