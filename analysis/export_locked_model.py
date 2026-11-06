"""Fit the published calibration (quadratic-basis ridge, kappa-optimised cut-offs) on the
full cohort and save it as a plain JSON file, so that `ecvrs.cli render` shows the grades of
the actual model rather than generic cut-offs. Every number in the file is inspectable.

    ECVRS_OUT_DIR=analysis_out python analysis/export_locked_model.py
"""
import os, sys, json, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecvrs.features.panel import build_panel, FEATURE_NAMES
from ecvrs.calibration.quadratic_ridge import train_quadratic_ridge_mapping

OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
DEST = Path(__file__).resolve().parents[1] / 'ecvrs' / 'models' / 'locked_model_adni_mci.json'
LOBES = [('Hippo_Lt', 5), ('Hippo_Rt', 5), ('Front', 4), ('Parietal', 4), ('Temporal', 4)]

df = pd.read_csv(OUT_DIR / 'cohort_merged.csv')
X = build_panel(df)
model = {'description': 'e-CVRS locked calibration: quadratic-basis ridge per region, '
                        'kappa-optimised ordinal cut-offs, fitted on ADNI MCI (N=%d).' % len(df),
         'created': datetime.date.today().isoformat(),
         'feature_names': FEATURE_NAMES,
         'design': '[x, x^2] for the 11 features (22 terms), z-scored with mu/sd below',
         'regions': {}}
for region, n in LOBES:
    m = train_quadratic_ridge_mapping(X, df[region].values, n, seed=42)
    model['regions'][region] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
DEST.parent.mkdir(parents=True, exist_ok=True)
json.dump(model, open(DEST, 'w'), indent=1, default=float)
print('saved', DEST)
