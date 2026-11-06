"""The 11-feature panel used by every regional calibration model.

Order (fixed; the locked model's coefficients refer to it):
  0-4   CSF fractions: MTA left, MTA right, frontal, parietal, temporal
  5-9   proxy volumes divided by total brain volume (TBV), same regions
  10    ventricular fraction (central ventricular CSF volume / TBV)
"""
import numpy as np

RATIO_COLS = ['Auto_Lt_Ratio', 'Auto_Rt_Ratio', 'Auto_Front_Ratio',
              'Auto_Parietal_Ratio', 'Auto_Temporal_Ratio']
VOL_COLS = ['Vol_Hippo_Lt', 'Vol_Hippo_Rt', 'Vol_Front', 'Vol_Parietal', 'Vol_Temporal']
FEATURE_NAMES = ['CSF ratio MTA-L', 'CSF ratio MTA-R', 'CSF ratio frontal', 'CSF ratio parietal',
                 'CSF ratio temporal', 'Volume MTA-L / TBV', 'Volume MTA-R / TBV',
                 'Volume frontal / TBV', 'Volume parietal / TBV', 'Volume temporal / TBV',
                 'Ventricular fraction']


def build_panel(rows):
    """rows: dict (one subject) or DataFrame with the raw feature columns -> (n, 11) array."""
    get = (lambda c: np.atleast_1d(np.asarray(rows[c], dtype=float)))
    tbv = get('Vol_TBV')
    tbv = np.where(tbv == 0, 1.0, tbv)
    cols = [get(c) for c in RATIO_COLS] + [get(v) / tbv for v in VOL_COLS] + [get('Vol_Ventricle') / tbv]
    return np.column_stack(cols)
