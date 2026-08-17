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

# ---- validation anchors vs published numbers ----
# These anchors compare the freshly merged cohort against the v2 published
# grade file. They are diagnostics for the original authors; skipped when that
# file is absent (e.g. in a fresh clone).
_anchor = DATA_DIR / 'e-CVRS_linear_multi_grades.csv'
if not _anchor.exists():
    print('anchor file not found; skipping reproduction checks')
    raise SystemExit(0)

from scipy.stats import spearmanr
pub = pd.read_csv(_anchor)  # seed-42 OLS grades
m = df.merge(pub[['RID','e_Atrophy_Sum']], on='RID')
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecvrs.stats.agreement import intraclass_correlation
icc = intraclass_correlation(m['Manual_Atrophy_Sum'].values, m['e_Atrophy_Sum'].values)
print(f'anchor ICC (expect 0.729): {icc:.4f}')
ok = m.FS_Hippo_Total.notna()
rho_m,_ = spearmanr(m.loc[ok,'Manual_Atrophy_Sum'], -m.loc[ok,'FS_Hippo_Total'])
rho_e,_ = spearmanr(m.loc[ok,'e_Atrophy_Sum'], -m.loc[ok,'FS_Hippo_Total'])
print(f'anchor FS rho manual (expect 0.472): {rho_m:.3f} | e-CVRS (expect 0.367): {rho_e:.3f}')
csf = m.dropna(subset=['ABETA','PTAU'])
print('CSF complete N (expect 321):', len(csf))
r1,_ = spearmanr(csf['e_Atrophy_Sum'], csf['ABETA']); r2,_ = spearmanr(csf['e_Atrophy_Sum'], csf['PTAU'])
r3,_ = spearmanr(csf['Manual_Atrophy_Sum'], csf['ABETA'])
print(f'anchor CSF: e-Ab42 (expect -0.284): {r1:.3f} | e-ptau (expect 0.047): {r2:.3f} | manual-Ab42 (expect -0.231): {r3:.3f}')
