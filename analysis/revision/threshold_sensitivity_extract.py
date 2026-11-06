"""Intensity-fraction sensitivity, step 1: features for all scans under each CSF fraction changed by
+/-0.05 (one at a time and all together) and head-mask fractions 0.10 and 0.14; one JSON per scan.
The "base" variant reproduces the published features.

    python analysis/revision/threshold_sensitivity_extract.py REPO SCAN_DIR OUT_DIR cohort_merged.csv
"""
import sys, os, re, glob, json, copy, numpy as np, pandas as pd, yaml
REPO, SCAN, OUT, COH = sys.argv[1:5]
sys.path.insert(0, REPO)
from ecvrs.io.readers import load_mri, standardize_orientation
from ecvrs.features.csf_ratio import extract_features
from ecvrs.features import proxy_volume as PV
CFG = yaml.safe_load(open(os.path.join(REPO, 'ecvrs/config/default.yaml')))
REG = {'mta': 'mta', 'front': 'frontal', 'par': 'parietal', 'temp': 'temporal'}
CSF_VARIANTS = {}
for k, r in REG.items():
    for d in (-0.05, 0.05):
        CSF_VARIANTS[f'{k}{d:+.2f}'] = {r: round(CFG[r]['tf'] + d, 2)}
for d in (-0.05, 0.05):
    CSF_VARIANTS[f'all{d:+.2f}'] = {r: round(CFG[r]['tf'] + d, 2) for r in REG.values()}

def com_frac(data, i98, frac):
    m = data > (frac * i98); z = np.argwhere(m)
    zl = int(z[:, 2].max() - 130)
    if zl > 0: m[:, :, :zl] = False
    return np.argwhere(m).mean(axis=0)

def feats(data, sp, i98, com, cfg, brain_frac=0.12):
    r, _ = extract_features(data, *com, sp, i98, cfg)
    old = PV.BRAIN_FRACTION; PV.BRAIN_FRACTION = brain_frac
    try: v = PV.extract_proxy_volumes(data, *com, sp, i98, cfg)
    finally: PV.BRAIN_FRACTION = old
    out = dict(r); out.update(v); return out

rids = set(pd.read_csv(COH).RID); n = 0
for f in sorted(glob.glob(os.path.join(SCAN, '*.hdr'))):
    m = re.search(r'_S_(\d+)_', os.path.basename(f))
    if not m or int(m.group(1)) not in rids or not os.path.exists(f[:-4] + '.img'): continue
    rid = int(m.group(1)); fn = os.path.join(OUT, f'{rid}.json')
    if os.path.exists(fn): continue
    data, aff, sp = load_mri(f); data = standardize_orientation(data, aff)
    i98 = np.percentile(data[::2, ::2, ::2], 98)
    com = tuple(com_frac(data, i98, 0.12))
    res = {'base': feats(data, sp, i98, com, CFG)}
    for name, ch in CSF_VARIANTS.items():
        cfg = copy.deepcopy(CFG)
        for reg, tf in ch.items(): cfg[reg]['tf'] = tf
        r, _ = extract_features(data, *com, sp, i98, cfg)
        res[name] = dict(res['base']); res[name].update(r)
    for hf in (0.10, 0.14):
        c2 = tuple(com_frac(data, i98, hf))
        res[f'head{hf:.2f}'] = feats(data, sp, i98, c2, CFG, brain_frac=hf)
    json.dump({'rid': rid, 'variants': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}}, open(fn, 'w')); n += 1
print('done now', n, 'total', len(os.listdir(OUT)))
