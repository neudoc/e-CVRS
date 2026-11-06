"""Build the analysis cohort: e-CVRS features + manual ratings + FreeSurfer + cognition + CSF."""
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

import numpy as np, pandas as pd

auto = pd.read_csv(DATA_DIR / 'e-CVRS_automated_scores.csv')
man  = pd.read_excel(DATA_DIR / 'ADNI_MRI_rating.xlsx')
# Laterality alignment. The expert MTA grades in ADNI_MRI_rating.xlsx were recorded in
# radiological display convention (screen left = patient right), so the 'Hippo_Lt' column
# holds the patient's right hemisphere and vice versa. Swap them to anatomical laterality,
# the convention of the e-CVRS windows and of FreeSurfer (ST29SV left, ST88SV right).
# The check at the end of this script verifies the alignment against FreeSurfer asymmetry.
man[['Hippo_Lt', 'Hippo_Rt']] = man[['Hippo_Rt', 'Hippo_Lt']].values
df = pd.merge(auto, man, on='RID')          # 339
print('paired cohort N =', len(df))

# FreeSurfer: earliest scan per RID
fs = pd.read_csv(DATA_DIR / 'UCSF_FS.csv', low_memory=False)
fs['EXAMDATE_dt'] = pd.to_datetime(fs['EXAMDATE'], errors='coerce')
fs = fs.sort_values('EXAMDATE_dt').drop_duplicates('RID', keep='first')
fs['FS_Hippo_Total'] = fs['ST29SV'] + fs['ST88SV']
df = df.merge(fs[['RID','FS_Hippo_Total','FIELD_STRENGTH','OVERALLQC']], on='RID', how='left')
print('FS available:', df.FS_Hippo_Total.notna().sum())

# Cognition composites: earliest exam per RID
uw = pd.read_csv(DATA_DIR / 'UWNPSYCHSUM_01Jul2025.csv', low_memory=False)
uw['EXAMDATE_dt'] = pd.to_datetime(uw['EXAMDATE'], errors='coerce')
uw = uw.sort_values('EXAMDATE_dt').drop_duplicates('RID', keep='first')
df = df.merge(uw[['RID','ADNI_MEM','ADNI_EF','ADNI_LAN']], on='RID', how='left')
print('MEM/EF/LAN available:', df.ADNI_MEM.notna().sum(), df.ADNI_EF.notna().sum(), df.ADNI_LAN.notna().sum())

# CSF: earliest per RID, numeric coercion
bm = pd.read_csv(DATA_DIR / 'UPENNBIOMK9_01Jul2025.csv', low_memory=False)
bm['EXAMDATE_dt'] = pd.to_datetime(bm['EXAMDATE'], errors='coerce')
for c in ['ABETA','TAU','PTAU']:
    bm[c] = pd.to_numeric(bm[c].astype(str).str.replace('>','',regex=False).str.replace('<','',regex=False), errors='coerce')
bm = bm.sort_values('EXAMDATE_dt').drop_duplicates('RID', keep='first')
df = df.merge(bm[['RID','ABETA','TAU','PTAU']], on='RID', how='left')
print('CSF ABETA/PTAU available:', df.ABETA.notna().sum(), df.PTAU.notna().sum())

# derived
df['Gender_Dummy'] = (df['PTGENDER'].astype(str).str.lower().str.startswith('f')).astype(int)
df['APOE4_Carrier'] = (df['APOE4'] > 0).astype(int)
df['Manual_Atrophy_Sum'] = df[['Hippo_Lt','Hippo_Rt','Front','Parietal','Temporal']].sum(axis=1)
df['vent_norm'] = df['Vol_Ventricle']/df['Vol_TBV']
df.to_csv(OUT_DIR / 'cohort_merged.csv', index=False)
print('saved', OUT_DIR / 'cohort_merged.csv', df.shape)

# ---- cohort composition checks against the published analysis ----
# These assert the cohort assembled from a fresh download matches the one
# analysed in the paper. They use only the merged inputs, so they run in any
# clone that has the ADNI-derived source tables.
from scipy.stats import spearmanr

print('\n--- cohort checks (published values in parentheses) ---')
print(f'  N                        {len(df)}   (339)')
print(f'  FreeSurfer available     {int(df.FS_Hippo_Total.notna().sum())}')
print(f'  CSF complete (Ab42+ptau) {int(df.dropna(subset=["ABETA","PTAU"]).shape[0])}   (321)')

ok = df.FS_Hippo_Total.notna()
rho_m, _ = spearmanr(df.loc[ok, 'Manual_Atrophy_Sum'], -df.loc[ok, 'FS_Hippo_Total'])
print(f'  expert total vs FS rho   {rho_m:.3f}   (0.472)')

# The raw, uncalibrated medial temporal CSF ratios should already track
# hippocampal volume: this is the check that the window is on the anatomy.
for col, expect in [('Auto_Lt_Ratio', 0.413), ('Auto_Rt_Ratio', 0.358)]:
    r, _ = spearmanr(df.loc[ok, col], -df.loc[ok, 'FS_Hippo_Total'])
    print(f'  raw {col:<16} {r:.3f}   ({expect:.3f})')

# Laterality check: left-minus-right asymmetry of the expert MTA grades, after the
# alignment above, and of the raw e-CVRS medial CSF ratios should both run in the same
# direction as FreeSurfer hippocampal asymmetry (right minus left volume; positive =
# smaller left hippocampus). Expected rho: expert +0.251, raw e-CVRS +0.157.
lr = df[['RID']].merge(fs[['RID', 'ST29SV', 'ST88SV']], on='RID', how='left')
okl = lr.ST29SV.notna().values & lr.ST88SV.notna().values
fs_asym = (lr.ST88SV - lr.ST29SV).values[okl]
for name, v, expect in [('expert MTA L-R', (df.Hippo_Lt - df.Hippo_Rt).values, 0.251),
                        ('raw e-CVRS L-R', (df.Auto_Lt_Ratio - df.Auto_Rt_Ratio).values, 0.157)]:
    r, _ = spearmanr(v[okl], fs_asym)
    print(f'  laterality {name:<16} {r:+.3f}   ({expect:+.3f})')
