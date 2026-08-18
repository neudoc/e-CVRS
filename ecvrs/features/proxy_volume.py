"""Box-based tissue and CSF volume proxies.

These complement the CSF-fraction features by giving each region a crude but
fully explicit size measurement: the number of voxels inside a named 3D box
whose intensity falls in the tissue range, times the voxel volume. No
segmentation is performed and no anatomical atlas is consulted; the boxes are
placed by physical millimetre offsets from the head centre of mass.

The medial temporal box shares the vertical extent of the hippocampus-anchored
MTA window (``mta.dz_center`` +/- ``mta.height_z``/2) and is centred medially
enough to track the hippocampal formation rather than lateral temporal cortex.
"""

import numpy as np

# Intensity gates, as fractions of the 98th-percentile intensity.
CSF_FRACTION = 0.25       # below this: CSF
TISSUE_CEILING = 0.95     # above this: bright non-parenchymal signal
BRAIN_FRACTION = 0.12     # crude head mask for total brain volume
NECK_CLIP_SLICES = 130    # axial slices retained below the vertex


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _box(data, x, y, z):
    return data[x[0]:x[1], y[0]:y[1], z[0]:z[1]]


def _tissue_volume(roi, lo, hi, vox_vol):
    if roi.size == 0:
        return 0.0
    return float(((roi >= lo) & (roi < hi)).sum() * vox_vol)


def extract_proxy_volumes(data, x_com, y_com, z_com, spacing, i98, cfg=None):
    """Return the proxy-volume panel, in mm^3.

    ``cfg`` is the loaded configuration dict. When omitted, the published
    hippocampus-anchored defaults are used, so the function stays callable
    from older scripts.
    """
    if cfg is None:
        cfg = {'mta': {'dz_center': -18, 'height_z': 25},
               'mta_volume': {'x_center': 24, 'width_x': 35,
                              'dy_center': 10, 'width_y': 20}}

    sx = abs(spacing[0])
    sy = spacing[1]
    sz = spacing[2]
    vox_vol = abs(spacing[0] * spacing[1] * spacing[2])

    csf_thresh = CSF_FRACTION * i98
    tissue_ceiling = TISSUE_CEILING * i98

    # ---- total brain volume (head mask with a fixed neck cutoff) ----
    brain_mask = data > (BRAIN_FRACTION * i98)
    idx = np.argwhere(brain_mask)
    if idx.size == 0:
        return {}
    z_top = idx[:, 2].max()
    z_limit = int(z_top - NECK_CLIP_SLICES)
    head_mask = brain_mask.copy()
    if z_limit > 0:
        head_mask[:, :, :z_limit] = False
    vol_tbv = float(head_mask.sum() * vox_vol)

    # ---- medial temporal boxes (hippocampus-anchored) ----
    m = cfg['mta']
    mv = cfg['mta_volume']

    z_center = int(z_com + m['dz_center'] / sz)
    half_z = int(m['height_z'] / sz) // 2
    mt_z = (_clamp(z_center - half_z, 0, data.shape[2]),
            _clamp(z_center + half_z, 0, data.shape[2]))

    half_y = int(mv['width_y'] / sy) // 2
    y_center = int(y_com + mv['dy_center'] / sy)
    mt_y = (_clamp(y_center - half_y, 0, data.shape[1]),
            _clamp(y_center + half_y, 0, data.shape[1]))

    half_x = int(mv['width_x'] / sx) // 2
    mt_vols = []
    for sign in (-1, 1):
        xc = int(x_com + sign * mv['x_center'] / sx)
        bx = (_clamp(xc - half_x, 0, data.shape[0]), _clamp(xc + half_x, 0, data.shape[0]))
        mt_vols.append(_tissue_volume(_box(data, bx, mt_y, mt_z),
                                      csf_thresh, tissue_ceiling, vox_vol))
    vol_hippo_lt, vol_hippo_rt = mt_vols

    # ---- central ventricular CSF volume ----
    v_x = (_clamp(int(x_com - 25 / sx), 0, data.shape[0]),
           _clamp(int(x_com + 25 / sx), 0, data.shape[0]))
    v_y = (_clamp(int(y_com - 30 / sy), 0, data.shape[1]),
           _clamp(int(y_com + 30 / sy), 0, data.shape[1]))
    v_z = (_clamp(int(z_com - 20 / sz), 0, data.shape[2]),
           _clamp(int(z_com + 20 / sz), 0, data.shape[2]))
    v_roi = _box(data, v_x, v_y, v_z)
    vol_ventricle = float((v_roi < csf_thresh).sum() * vox_vol) if v_roi.size else 0.0

    # ---- lobar tissue volume proxies ----
    def lobe(x_mm, y_mm, z_mm):
        bx = (_clamp(int(x_com + x_mm[0] / sx), 0, data.shape[0]),
              _clamp(int(x_com + x_mm[1] / sx), 0, data.shape[0]))
        by = (_clamp(int(y_com + y_mm[0] / sy), 0, data.shape[1]),
              _clamp(int(y_com + y_mm[1] / sy), 0, data.shape[1]))
        bz = (_clamp(int(z_com + z_mm[0] / sz), 0, data.shape[2]),
              _clamp(int(z_com + z_mm[1] / sz), 0, data.shape[2]))
        return _tissue_volume(_box(data, bx, by, bz), csf_thresh, tissue_ceiling, vox_vol)

    vol_front = lobe((-40, 40), (10, 60), (20, 60))
    vol_parietal = lobe((-40, 40), (-60, -10), (25, 65))
    vol_temporal = (lobe((-60, -35), (-30, 30), (-20, 20))
                    + lobe((35, 60), (-30, 30), (-20, 20)))

    return {
        'Vol_Hippo_Lt': vol_hippo_lt,
        'Vol_Hippo_Rt': vol_hippo_rt,
        'Vol_Front': vol_front,
        'Vol_Parietal': vol_parietal,
        'Vol_Temporal': vol_temporal,
        'Vol_Ventricle': vol_ventricle,
        'Vol_TBV': vol_tbv,
    }
