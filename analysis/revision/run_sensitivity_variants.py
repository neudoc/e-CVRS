"""Window-extent and intensity-fraction sensitivity, step 2 (Supplementary Tables S5 and S6).

Builds one cohort table per variant from the step-1 JSON files, runs the full benchmark
(5-fold x 8 seeds) and the primary-seed analysis for each, and writes a summary table.

    python analysis/revision/run_sensitivity_variants.py REPO OUT_ROOT \
        --cohort cohort_merged.csv --window SLICE_JSON_DIR --threshold THRESH_JSON_DIR \
        [--seq rev_case_meta_seq_orientation.csv] [--build-only]

--cohort is the analysis cohort written by analysis/merge_cohort.py. --seq (columns RID, seq) adds
the Kruskal-Wallis test of the total error across acquisition sequences.
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('out_root')
ap.add_argument('--cohort', required=True)
ap.add_argument('--window'); ap.add_argument('--threshold'); ap.add_argument('--seq')
ap.add_argument('--build-only', action='store_true')
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, 'analysis', 'revision'))

base = pd.read_csv(a.cohort)
FEATS = ['Auto_Lt_Ratio', 'Auto_Rt_Ratio', 'Auto_Front_Ratio', 'Auto_Parietal_Ratio', 'Auto_Temporal_Ratio',
         'Vol_Hippo_Lt', 'Vol_Hippo_Rt', 'Vol_Front', 'Vol_Parietal', 'Vol_Temporal', 'Vol_Ventricle', 'Vol_TBV']
variants = {}

# Window extent (Table S5): (CSF-ratio dy range, volume-box dy range), mm from the COM.
if a.window:
    from window_assemble import assemble
    W = {'win_m5_p25': (-5, 25, 0, 20), 'win_m5_p20': (-5, 20, 0, 20), 'win_m5_p15': (-5, 15, 0, 20),
         'win_m5_p10': (-5, 10, 0, 20), 'win_m5_p15_vol': (-5, 15, -5, 15)}
    for name, (y0, y1, v0, v1) in W.items():
        f = assemble(a.window, y0, y1, v0, v1).set_index('RID').loc[base.RID]
        x = base.copy()
        for col in ['Auto_Lt_Ratio', 'Auto_Rt_Ratio', 'Vol_Hippo_Lt', 'Vol_Hippo_Rt']:
            x[col] = f[col].values
        variants[name] = x

# Intensity fractions (Table S6).
if a.threshold:
    D = {}
    for fn in glob.glob(os.path.join(a.threshold, '*.json')):
        d = json.load(open(fn)); D[d['rid']] = d['variants']
    for name in next(iter(D.values())):
        x = base.copy()
        for col in FEATS:
            x[col] = [D[r][name][col] for r in base.RID]
        x['vent_norm'] = x.Vol_Ventricle / x.Vol_TBV
        variants['thr_' + name] = x

for name, x in variants.items():
    d = os.path.join(a.out_root, name); os.makedirs(d, exist_ok=True)
    x.to_csv(os.path.join(d, 'cohort_merged.csv'), index=False)
print('built', len(variants), 'variants')
if a.build_only:
    sys.exit(0)

for name in variants:
    d = os.path.join(a.out_root, name)
    if os.path.exists(os.path.join(d, 'final_results.json')):
        continue
    env = dict(os.environ, ECVRS_DATA_DIR=d, ECVRS_OUT_DIR=d)
    for script in ('benchmark.py', 'final_analysis.py'):
        with open(os.path.join(d, f'log_{script[:-3]}.txt'), 'w') as log:
            subprocess.run([sys.executable, os.path.join(a.repo, 'analysis', script)],
                           env=env, cwd=a.repo, stdout=log, stderr=subprocess.STDOUT, check=True)

seq = pd.read_csv(a.seq)[['RID', 'seq']] if a.seq else None
rows, ref = [], {}
for name in variants:
    d = os.path.join(a.out_root, name)
    c = pd.read_csv(os.path.join(d, 'cohort_merged.csv'))
    e = pd.read_csv(os.path.join(d, 'oof_seed42_full_quadratic_ridge.csv'))
    b = pd.read_csv(os.path.join(d, 'benchmark_results.csv'))
    q = b[b.candidate == 'full_quadratic_ridge'].mean(numeric_only=True)
    diff = pd.DataFrame({'RID': c.RID, 'diff': e.e_sum.values - c.Manual_Atrophy_Sum.values})
    disc = set(diff.RID[diff['diff'].abs() >= 5])
    group = name.split('_')[0]
    ref.setdefault(group, disc)                      # first variant of each group = published setting
    r = dict(variant=name, icc_total=q.icc_total, kappa_mta_lt=q.kappa_Hippo_Lt, kappa_mta_rt=q.kappa_Hippo_Rt,
             kappa_front=q.kappa_Front, kappa_parietal=q.kappa_Parietal, kappa_temporal=q.kappa_Temporal,
             fs_rho=q.fs_rho_total, cog_mean_F=q.cog_mean_F, csf_mean_abs=q.csf_mean_abs,
             n_discordant=len(disc), shared_with_published=len(disc & ref[group]))
    if seq is not None:
        from scipy.stats import kruskal
        m = diff.merge(seq, on='RID')
        r['seq_kw_p'] = kruskal(*[g['diff'] for _, g in m.groupby('seq')]).pvalue
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(a.out_root, 'sensitivity_summary.csv'), index=False)
print(out.round(3).to_string(index=False))
