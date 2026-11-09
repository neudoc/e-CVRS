"""Main-text Figures 1-4 for the published e-CVRS analysis.

Reads final_results.json and the seed-42 out-of-fold grades produced by
analysis/final_analysis.py and writes the figures into ECVRS_FIG_DIR.

Figure 5 (the anatomical window overlay) is produced by the pipeline itself:
    python -m ecvrs.cli render --scan <subject.hdr> --output_dir case_out
"""

import os
from pathlib import Path

OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
FIG_DIR = Path(os.environ.get("ECVRS_FIG_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

o = json.load(open(OUT_DIR / 'final_results.json'))
df = pd.read_csv(OUT_DIR / 'cohort_merged.csv')
e42 = pd.read_csv(OUT_DIR / 'oof_seed42_full_quadratic_ridge.csv')

C_ECVRS = '#0072B2'; C_MAN = '#555555'; C_FS = '#D55E00'; C_ACC = '#009E73'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#E5E5E5',
                     'grid.linewidth': 0.6, 'axes.axisbelow': True})
FD = str(FIG_DIR)

# ============ Figure 1: pipeline and the two evaluation paths ============
fig, ax = plt.subplots(figsize=(11, 7.4)); ax.axis('off'); ax.grid(False)
def box(x, y, w, h, text, title=None, fc='#F2F6FA', ec='#0072B2', fs=8.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.010', fc=fc, ec=ec, lw=1.2))
    if title:
        ax.text(x + w/2, y + h - 0.022, title, ha='center', va='top', fontsize=fs + 0.6, fontweight='bold')
        ax.text(x + w/2, y + (h - 0.045)/2, text, ha='center', va='center', fontsize=fs)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal')
def arrow(x1, y1, x2, y2, c='#333'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, color=c, lw=1.1))
def letter(y, t):
    ax.text(-0.03, y, t, fontsize=12, fontweight='bold', va='center')
ax.set_xlim(-0.04, 1); ax.set_ylim(0, 1)
ax.text(0.48, 1.0, 'e-CVRS: feature extraction, calibration, and the two evaluation paths',
        ha='center', va='top', fontsize=10.5, fontweight='bold')
# A. feature extraction (fixed rules)
y0, h0 = 0.80, 0.13
letter(y0 + h0/2, 'A')
box(0.01, y0, 0.215, h0, 'Baseline 3D T1 MRI\n(ADNI MCI, N=339)\nno registration or segmentation')
box(0.265, y0, 0.215, h0, 'Center-of-mass anchoring\nneck stripping;\nphysical-mm offsets')
box(0.52, y0, 0.215, h0, 'Hippocampus-anchored MTA\nwindows + lobar windows\n→ 11 named features')
box(0.775, y0, 0.215, h0, 'Explanatory overlay per scan\nwindows, CSF masks,\nfeatures, grades', ec=C_ACC, fc='#F0FAF6')
for x in (0.225, 0.48, 0.735): arrow(x, y0 + h0/2, x + 0.04, y0 + h0/2)
ax.text(0.37, 0.772, 'fixed extraction rules; nothing is fitted', ha='center', fontsize=7.8, style='italic', color='#444')
# B. calibration of one candidate
y1, h1 = 0.60, 0.15
letter(y1 + h1/2, 'B')
box(0.08, y1, 0.84, h1, 'per region: features → candidate model (e.g. quadratic ridge [x, x²], λ chosen by inner 4-fold CV)\n'
    '→ grade cut-offs maximizing quadratic-weighted κ → grades for the held-out fold (5-fold outer CV × 8 seeds)',
    title='Calibration of one candidate model (always inside the outer training fold)', ec=C_ECVRS, fc='#EAF2FB')
arrow(0.6275, y0, 0.6275, y1 + h1)
# C. two evaluation paths
y2, h2 = 0.255, 0.30
letter(y2 + h2/2, 'C')
box(0.01, y2, 0.465, h2,
    'All five candidates evaluated out-of-fold;\n'
    'composite rank of FreeSurfer ρ, cognitive F and CSF ρ\n'
    'computed on all participants and eight seeds\n'
    '→ quadratic ridge chosen once and reported\n\n'
    'The choice uses the reported outcomes (not nested)\n'
    'Total ICC 0.748 (primary split); 0.734 (eight-seed mean)',
    title='Primary analysis (fixed candidate)', ec='#555555', fc='#F7F7F7')
box(0.525, y2, 0.465, h2,
    'Inside each outer training fold only:\n'
    'five candidates evaluated by inner 5-fold CV,\n'
    'composite rank on training participants,\n'
    'chosen candidate refitted and applied to the held-out fold\n\n'
    'Selection repeated inside the folds (still internal validation)\n'
    'Total ICC 0.736 (primary split); 0.720 (eight-seed mean)',
    title='Selection-honest analysis (nested)', ec=C_ECVRS, fc='#EAF2FB')
arrow(0.30, y1, 0.24, y2 + h2); arrow(0.70, y1, 0.76, y2 + h2)
# D. outputs
y3, h3 = 0.02, 0.19
letter(y3 + h3/2, 'D')
box(0.01, y3, 0.30, h3, 'MTA left/right (0–4)\nfrontal / parietal / temporal (0–3)\ntotal (0–17)', title='Out-of-fold grades')
box(0.355, y3, 0.635, h3, 'agreement with the expert CVRS rating (κw, ICC, Bland–Altman)\n'
    'FreeSurfer hippocampal volume (ρ); six cognitive outcomes (incremental F)\n'
    'CSF Aβ42 and p-tau181 (ρ)', title='Internal validation in one ADNI cohort', ec=C_FS, fc='#FDF3EC')
arrow(0.24, y2, 0.16, y3 + h3); arrow(0.76, y2, 0.67, y3 + h3); arrow(0.31, y3 + h3/2, 0.355, y3 + h3/2)
plt.savefig(f'{FD}/Figure1_pipeline.png', dpi=300, bbox_inches='tight'); plt.close()

# ============ Figure 2: agreement ============
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
regs = [('Hippo_Lt', 'Left hippocampus MTA'), ('Hippo_Rt', 'Right hippocampus MTA'),
        ('Front', 'Frontal'), ('Parietal', 'Parietal'), ('Temporal', 'Temporal')]
ys = np.arange(len(regs))[::-1]
for y, (rk, lab) in zip(ys, regs):
    v = o['table2'][rk]
    ax.errorbar(v['kappa'], y, xerr=[[v['kappa'] - v['lo']], [v['hi'] - v['kappa']]],
                fmt='o', color=C_ECVRS, capsize=3, ms=6)
    ax.text(v['hi'] + 0.02, y, f"{v['kappa']:.2f}", va='center', fontsize=8.5)
tv = o['table2']['Total']
ax.errorbar(tv['icc'], -1.2, xerr=[[tv['icc'] - tv['lo']], [tv['hi'] - tv['icc']]],
            fmt='s', color=C_FS, capsize=3, ms=7)
ax.text(tv['hi'] + 0.02, -1.2, f"{tv['icc']:.2f}", va='center', fontsize=8.5, fontweight='bold')
ax.set_yticks(list(ys) + [-1.2]); ax.set_yticklabels([l for _, l in regs] + ['Total score ICC(2,1)'])
ax.axvline(0.4, color='#BBB', lw=0.8, ls=':'); ax.axvline(0.6, color='#BBB', lw=0.8, ls=':')
ax.set_xlim(0.3, 0.9); ax.set_xlabel('Quadratic-weighted κ / ICC (95% bootstrap CI)')
ax.set_title('A. Agreement with expert manual ratings', loc='left', fontsize=10, fontweight='bold')
ax = axes[1]
m = df['Manual_Atrophy_Sum'].values; e = e42['e_sum'].values
mean = (m + e) / 2; dif = e - m
jx = np.random.RandomState(1).uniform(-0.18, 0.18, len(m)); jy = np.random.RandomState(2).uniform(-0.18, 0.18, len(m))
ax.scatter(mean + jx, dif + jy, s=14, alpha=0.45, color=C_ECVRS, edgecolor='none')
bias = tv['bias']
ax.axhline(bias, color=C_FS, lw=1.4, label=f'Bias {bias:.2f}')
ax.axhline(tv['loa_lo'], color=C_FS, lw=1, ls='--', label=f"95% LoA {tv['loa_lo']:.2f} / {tv['loa_hi']:.2f}")
ax.axhline(tv['loa_hi'], color=C_FS, lw=1, ls='--')
ax.set_xlabel('Mean of automated and manual total score'); ax.set_ylabel('Automated − manual')
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.set_title('B. Bland–Altman, total atrophy score (0–17)', loc='left', fontsize=10, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FD}/Figure2_agreement.png', dpi=300, bbox_inches='tight'); plt.close()

# ============ Figure 3: cognitive forest ============
t3 = o['table3']
ocs = [('MMSE', 'MMSE'), ('ADAS13', 'ADAS-Cog-13'), ('CDRSB', 'CDR-SB'),
       ('ADNI_MEM', 'ADNI-MEM'), ('ADNI_EF', 'ADNI-EF'), ('ADNI_LAN', 'ADNI-LAN')]
flip = {'MMSE': -1, 'ADAS13': 1, 'CDRSB': 1, 'ADNI_MEM': -1, 'ADNI_EF': -1, 'ADNI_LAN': -1}
sds = {k: df[k].std() for k, _ in ocs}
preds = [('manual', 'Manual CVRS', C_MAN), ('ecvrs', 'e-CVRS', C_ECVRS), ('fs', 'FreeSurfer hippocampus', C_FS)]
fig, ax = plt.subplots(figsize=(7.6, 9.2))
GAP = 2.2
yy = 0; yticks = []; ylabs = []; sep_ys = []
for gi, (k, lab) in enumerate(ocs):
    for j, (pk, plab, col) in enumerate(preds):
        r = t3[k][pk]
        b = r['beta'] * flip[k] / sds[k]; se = r['se'] / sds[k]
        sig = r['p_bh'] < 0.05
        ax.errorbar(b, yy, xerr=1.96 * se, fmt='o' if pk == 'ecvrs' else ('s' if pk == 'manual' else 'D'),
                    color=col, mfc=col if sig else 'white', capsize=2.5, ms=6.5, lw=1.3)
        yy -= 1
    yticks.append(yy + 2); ylabs.append(lab)
    if gi < len(ocs) - 1:
        sep_ys.append(yy - GAP / 2 + 0.5)
    yy -= GAP
for sy in sep_ys:
    ax.axhline(sy, color='#DDDDDD', lw=0.8)
ax.axvline(0, color='#999', lw=0.9)
ax.set_yticks(yticks); ax.set_yticklabels(ylabs, fontsize=10)
ax.set_ylim(yy + GAP - 0.6, 1.0)
ax.set_xlabel('Standardized β (95% CI), oriented so positive = more atrophy → worse cognition')
hs = [mlines.Line2D([], [], color=c, marker=mk, ls='None', ms=6, label=l)
      for (pk, l, c), mk in zip(preds, ['s', 'o', 'D'])]
ax.legend(handles=hs, frameon=False, fontsize=9, loc='lower right')
ax.set_title('Covariate-adjusted association with cognition (filled = BH-adjusted p<0.05)', loc='left', fontsize=10, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FD}/Figure3_cognition.png', dpi=300, bbox_inches='tight'); plt.close()

# ============ Figure 4: CSF ============
t4 = o['table4']
fig, ax = plt.subplots(figsize=(7.2, 3.6))
groups = [('abeta', 'CSF Aβ42'), ('ptau', 'CSF p-tau181')]
yy = 0; yticks = []; ylabs = []
for bmk, glab in groups:
    for pk, plab, col in preds:
        v = t4[f'{pk}_{bmk}']
        rho = v['rho']; lo = v['lo']; hi = v['hi']
        if pk == 'fs':
            rho, lo, hi = -rho, -v['hi'], -v['lo']
        sig = v['p'] < 0.05
        ax.errorbar(rho, yy, xerr=[[rho - lo], [hi - rho]], fmt='o' if pk == 'ecvrs' else ('s' if pk == 'manual' else 'D'),
                    color=col, mfc=col if sig else 'white', capsize=2.5, ms=6)
        yy -= 1
    yticks.append(yy + 2); ylabs.append(glab); yy -= 0.8
ax.axvline(0, color='#999', lw=0.9)
ax.set_yticks(yticks); ax.set_yticklabels(ylabs)
ax.set_xlabel('Spearman ρ (95% bootstrap CI), atrophy direction (N=321)')
ax.legend(handles=hs, frameon=False, fontsize=8.5, loc='lower right')
ax.set_title('Association with CSF biomarkers (filled = p<0.05)', loc='left', fontsize=9.5, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FD}/Figure4_csf.png', dpi=300, bbox_inches='tight'); plt.close()
print('figures 1-4 done:', os.listdir(FD))
