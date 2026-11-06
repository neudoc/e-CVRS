"""Assemble MTA features for a given anterior-posterior range from the per-slice profiles."""
import json, os, glob, numpy as np, pandas as pd
def assemble(sdir, dy0, dy1, vol_dy0=0, vol_dy1=20):
    rows = []
    for f in glob.glob(os.path.join(sdir, '*.json')):
        d = json.load(open(f)); yc = d['com'][1]; sy = d['sp'][1]
        r = {'RID': d['rid']}
        for s, col in (('lh', 'Lt'), ('rh', 'Rt')):
            r[f'Auto_{col}_Ratio'] = float(np.mean([d['ratio'][s][str(k)] for k in range(dy0, dy1 + 1)]))
            # volume box: y voxel range [int(yc+c/sy)-half, int(yc+c/sy)+half), as in proxy_volume
            c = (vol_dy0 + vol_dy1) / 2; w = vol_dy1 - vol_dy0
            half = int(w / sy) // 2; ycen = int(yc + c / sy)
            a, b = ycen - half - d['ylo'], ycen + half - d['ylo']
            r[f'Vol_Hippo_{col}'] = float(sum(d['vol'][s][a:b]))
        rows.append(r)
    return pd.DataFrame(rows)
