"""Window-extent sensitivity, step 1: per-slice medial temporal CSF ratios (dy = -15..+35 mm from the
COM) and per-row tissue volumes in the MTA volume box, one JSON per scan, so that any anterior-posterior
range can be assembled offline (window_assemble.py). Assembling -5..+25 mm reproduces the published
Auto_Lt/Rt_Ratio and Vol_Hippo_Lt/Rt exactly.

    python analysis/revision/window_profile_extract.py REPO SCAN_DIR OUT_DIR cohort_merged.csv
"""
import sys, os, re, glob, json, numpy as np, pandas as pd, yaml
REPO, SCAN, OUT, COH = sys.argv[1:5]
sys.path.insert(0, REPO)
from ecvrs.io.readers import load_mri, standardize_orientation
from ecvrs.preprocess.com import calculate_com
from ecvrs.features import proxy_volume as PV
CFG = yaml.safe_load(open(os.path.join(REPO, 'ecvrs/config/default.yaml')))
m, mv = CFG['mta'], CFG['mta_volume']
cl = lambda v, lo, hi: max(lo, min(hi, v))
rids = set(pd.read_csv(COH).RID)
smap = {}
for f in sorted(glob.glob(os.path.join(SCAN, '*.hdr'))):
    r = re.search(r'_S_(\d+)_', os.path.basename(f))
    if r and int(r.group(1)) in rids and os.path.exists(f[:-4] + '.img'): smap[int(r.group(1))] = f
n = 0
for rid, f in smap.items():
    fn = os.path.join(OUT, f'{rid}.json')
    if os.path.exists(fn): continue
    data, aff, sp = load_mri(f); data = standardize_orientation(data, aff)
    i98 = np.percentile(data[::2, ::2, ::2], 98)
    xc, yc, zc, ok, _ = calculate_com(data, i98)
    sx, sy, sz = abs(sp[0]), sp[1], sp[2]; vox = abs(sp[0] * sp[1] * sp[2])
    th = m['tf'] * i98
    zcen = int(zc + m['dz_center'] / sz); hz = int(m['height_z'] / sz) // 2
    z0, z1 = cl(zcen - hz, 0, data.shape[2]), cl(zcen + hz, 0, data.shape[2])
    bands = {}
    for side, sg in (('lh', -1), ('rh', 1)):
        xa, xb = int(xc + sg * m['x_inner'] / sx), int(xc + sg * m['x_outer'] / sx)
        lo, hi = sorted((xa, xb)); bands[side] = (cl(lo, 0, data.shape[0]), cl(hi, 0, data.shape[0]))
    ratio = {s: {} for s in bands}
    for dy in range(-15, 36):
        y = cl(int(yc + dy / sy), 0, data.shape[1] - 1)
        for s, (x0, x1) in bands.items():
            roi = data[x0:x1, y, z0:z1]; ratio[s][dy] = float((roi < th).mean()) if roi.size else 0.0
    csf_t, ceil_t = PV.CSF_FRACTION * i98, PV.TISSUE_CEILING * i98
    hx = int(mv['width_x'] / sx) // 2
    ylo, yhi = cl(int(yc - 25 / sy), 0, data.shape[1]), cl(int(yc + 45 / sy), 0, data.shape[1])
    vol = {}
    for s, sg in (('lh', -1), ('rh', 1)):
        xcb = int(xc + sg * mv['x_center'] / sx)
        bx = (cl(xcb - hx, 0, data.shape[0]), cl(xcb + hx, 0, data.shape[0]))
        roi = data[bx[0]:bx[1], ylo:yhi, z0:z1]
        vol[s] = (((roi >= csf_t) & (roi < ceil_t)).sum(axis=(0, 2)) * vox).tolist()
    json.dump(dict(rid=rid, com=[float(xc), float(yc), float(zc)], sp=[float(sx), float(sy), float(sz)],
                   ylo=int(ylo), ratio=ratio, vol=vol), open(fn, 'w')); n += 1
print('done now', n, 'total', len(os.listdir(OUT)))
