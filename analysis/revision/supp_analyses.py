"""Supplementary analyses for the revision: confusion matrices, sub-scale Bland-Altman,
sequence-stratified error, grade distributions, cognitive-outcome correlation matrix, and
calibration stability (lambda, thresholds, coefficient signs) across the 40 outer folds."""
import sys, json
REPO = sys.argv[1]; SEQ = sys.argv[2]; FIGDIR = sys.argv[3]
sys.path.insert(0, REPO + '/analysis'); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import benchmark as B
from ecvrs.calibration.ordinal import train_kappa_optimized_mapping

OUT = B.OUT_DIR
df = pd.read_csv(OUT / 'cohort_merged.csv'); X = B.build_X(df)
e = pd.read_csv(OUT / 'oof_seed42_full_quadratic_ridge.csv')
seq = pd.read_csv(SEQ)[['RID', 'seq']]
REG = [r for r, _ in B.LOBES]; NC = dict(B.LOBES)
NAMES = {'Hippo_Lt': 'MTA left', 'Hippo_Rt': 'MTA right', 'Front': 'Frontal', 'Parietal': 'Parietal', 'Temporal': 'Temporal'}
res = {}

# 1. confusion matrices + grade distributions
fig, axs = plt.subplots(1, 5, figsize=(17, 3.9))
cm_all = {}
for ax, r in zip(axs, REG):
    n = NC[r]; cm = np.zeros((n, n), int)
    for t, p in zip(df[r].astype(int), e[r].astype(int)): cm[t, p] += 1
    cm_all[r] = cm.tolist()
    ax.imshow(cm, cmap='Blues')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > cm.max() * 0.55 else 'black')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel('e-CVRS grade'); ax.set_ylabel('Expert grade'); ax.set_title(NAMES[r], fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/FigureS1_confusion.png', dpi=300); plt.close()
res['confusion'] = cm_all
res['expert_grade_distribution'] = {r: df[r].value_counts().sort_index().to_dict() for r in REG}
res['ecvrs_grade_distribution'] = {r: e[r].value_counts().sort_index().to_dict() for r in REG}

# 2. sub-scale Bland-Altman + sequence-stratified total error
m = df[['RID']].copy()
m['man_h'] = df.Hippo_Lt + df.Hippo_Rt; m['auto_h'] = e.Hippo_Lt + e.Hippo_Rt
m['man_c'] = df.Front + df.Parietal + df.Temporal; m['auto_c'] = e.Front + e.Parietal + e.Temporal
m['man_t'] = df.Manual_Atrophy_Sum; m['auto_t'] = e.e_sum
m = m.merge(seq, on='RID', how='left')
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
rng = np.random.RandomState(0)
for ax, (a, b, ttl) in zip(axs[:2], [('man_h', 'auto_h', 'A. Hippocampal sub-scale (L+R, 0-8)'),
                                     ('man_c', 'auto_c', 'B. Cortical sub-scale (F+P+T, 0-9)')]):
    mean = (m[a] + m[b]) / 2; d = m[b] - m[a]
    bias, sd = d.mean(), d.std()
    ax.scatter(mean + rng.uniform(-.15, .15, len(m)), d + rng.uniform(-.15, .15, len(m)), s=12, alpha=.5, color='#0072B2')
    ax.axhline(bias, color='#D55E00', lw=2, label=f'Bias {bias:+.2f}')
    for v in (bias - 1.96 * sd, bias + 1.96 * sd): ax.axhline(v, color='#D55E00', ls='--', lw=1.2)
    ax.plot([], [], color='#D55E00', ls='--', label=f'95% LoA {bias-1.96*sd:.2f} / {bias+1.96*sd:.2f}')
    ax.set_xlabel('Mean of e-CVRS and expert'); ax.set_ylabel('e-CVRS - expert'); ax.set_title(ttl, fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)
    res[f'BA_{a}'] = dict(bias=float(bias), loa_lo=float(bias - 1.96 * sd), loa_hi=float(bias + 1.96 * sd))
ax = axs[2]
order = ['MPRAGE', 'MPRAGE_GRAPPA2', 'MPRAGE_SENSE2']
dd = [(m.loc[m.seq == s, 'auto_t'] - m.loc[m.seq == s, 'man_t']).values for s in order]
ax.boxplot(dd, widths=.5, showfliers=False)
for i, v in enumerate(dd):
    ax.scatter(np.full(len(v), i + 1) + rng.uniform(-.18, .18, len(v)), v + rng.uniform(-.12, .12, len(v)), s=9, alpha=.45, color='#0072B2')
ax.axhline(0, color='grey', lw=1); ax.axhline(5, color='#D55E00', ls=':', lw=1); ax.axhline(-5, color='#D55E00', ls=':', lw=1)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f'MPRAGE\n(n={len(dd[0])})', f'MPRAGE GRAPPA2\n(n={len(dd[1])})', f'MPRAGE SENSE2\n(n={len(dd[2])})'])
ax.set_ylabel('Total score: e-CVRS - expert'); ax.set_title('C. Total-score error by acquisition sequence', fontweight='bold', fontsize=11)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/FigureS2_agreement_error.png', dpi=300); plt.close()
res['seq_error'] = {s: dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std()), pct_ge5=float(100 * (np.abs(v) >= 5).mean())) for s, v in zip(order, dd)}

# 3. cognitive-outcome intercorrelation
cog = df[B.COGS].copy(); cog['ADAS13'] = -cog['ADAS13']; cog['CDRSB'] = -cog['CDRSB']
rho = cog.corr(method='spearman')
res['cognition_spearman_oriented'] = rho.round(3).to_dict()
off = rho.values[np.triu_indices(6, 1)]
res['cognition_rho_range'] = [float(off.min()), float(off.max()), float(np.median(np.abs(off)))]

# 4. calibration stability across 40 outer folds (published model)
lam_rows, thr_rows, sign_rows = [], [], []
for s in B.SEEDS:
    np.random.seed(s); idx = np.arange(len(df)); np.random.shuffle(idx); folds = np.array_split(idx, 5)
    for k in range(5):
        tr = np.hstack([folds[j] for j in range(5) if j != k])
        for r, n in B.LOBES:
            y = df[r].values.astype(float)[tr]
            _, lam = B.inner_select(X[tr], y, ['quadratic'], B.LAMBDAS, s)
            D = B.make_design(X[tr], 'quadratic'); mu, sd = B.zstats(D); Z = (D - mu) / sd
            b0, b = B.ridge_fit(Z, y, lam)
            thr = train_kappa_optimized_mapping(b0 + Z @ b, y, n)
            lam_rows.append(dict(seed=s, fold=k, region=r, lam=lam))
            thr_rows.append(dict(seed=s, fold=k, region=r, **{f'cut{i+1}': float(t) for i, t in enumerate(thr)}))
            sign_rows.append(dict(seed=s, fold=k, region=r, **{f'b{i}': float(v) for i, v in enumerate(b)}))
lam = pd.DataFrame(lam_rows); thr = pd.DataFrame(thr_rows); sg = pd.DataFrame(sign_rows)
res['lambda_counts'] = {r: lam[lam.region == r].lam.value_counts().sort_index().to_dict() for r in REG}
res['threshold_mean_sd'] = {r: {c: [float(thr.loc[thr.region == r, c].mean()), float(thr.loc[thr.region == r, c].std())]
                                for c in thr.columns if c.startswith('cut') and thr.loc[thr.region == r, c].notna().all()} for r in REG}
FEATS = ['CSF MTA L', 'CSF MTA R', 'CSF frontal', 'CSF parietal', 'CSF temporal', 'Vol MTA L', 'Vol MTA R', 'Vol frontal',
         'Vol parietal', 'Vol temporal', 'Ventricular frac']
FEATS2 = FEATS + [f + '^2' for f in FEATS]
stab = {}
for r in REG:
    t = sg[sg.region == r][[f'b{i}' for i in range(22)]].values
    med = np.median(np.abs(t), axis=0); top = np.argsort(-med)[:5]
    stab[r] = [dict(term=FEATS2[i], median_abs_coef=float(med[i]),
                    sign_consistency_pct=float(100 * max((t[:, i] > 0).mean(), (t[:, i] < 0).mean()))) for i in top]
res['top_coefficients'] = stab
lam.to_csv(OUT / 'rev_stability_lambda.csv', index=False); thr.to_csv(OUT / 'rev_stability_thresholds.csv', index=False)
sg.to_csv(OUT / 'rev_stability_coefficients.csv', index=False)
json.dump(res, open(OUT / 'rev_supplementary.json', 'w'), indent=1, default=float)
print(json.dumps({k: res[k] for k in ['expert_grade_distribution', 'seq_error', 'cognition_rho_range', 'lambda_counts', 'top_coefficients', 'BA_man_h', 'BA_man_c']}, indent=1, default=float)[:6000])
