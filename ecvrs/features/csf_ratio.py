"""Transparent CSF-fraction features for the e-CVRS subscales.

Every feature is the fraction of voxels inside an explicitly bounded window
whose intensity falls below a fixed fraction of the scan's 98th-percentile
intensity. There is no registration, no segmentation, and no learned image
representation: the number that drives each grade can be recomputed by hand
from the window bounds and the threshold printed alongside it.

Geometry (see ``ecvrs/config/default.yaml``):

* Medial temporal window -- a bilateral medial band, ``x_inner``-``x_outer`` mm
  lateral to the midline and ``height_z`` mm tall, centred ``dz_center`` mm
  below the centre of mass. The CSF fraction is averaged over every coronal
  slice from ``dy_start`` to ``dy_end`` mm, i.e. from the hippocampal tail
  forward to the head, so the feature integrates atrophy along the hippocampal
  long axis instead of sampling a single arbitrary slice.
* Lobar windows -- averaged over ``n_slices`` adjacent axial slices centred on
  the window plane, the cortical analogue of the multi-slice medial temporal
  sampling.
"""

import numpy as np


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _window_ratio(plane, x0, x1, z0, z1, thresh):
    """CSF fraction inside an axis-aligned patch of a 2D slice."""
    roi = plane[x0:x1, z0:z1]
    if roi.size == 0:
        return 0.0
    return float((roi < thresh).sum() / roi.size)


def extract_medial_temporal_ratios(data, x_com, y_com, z_com, spacing, i98, cfg):
    """Left/right medial temporal CSF fractions, averaged along the long axis.

    Returns ``(ratio_left, ratio_right, metadata)``. The metadata carries the
    exact voxel bounds used, so the overlay renderer and the QC checks describe
    the same window that produced the number.
    """
    m = cfg['mta']
    thresh = m['tf'] * i98

    sx = abs(spacing[0])
    sy = spacing[1]
    sz = spacing[2]

    # Vertical extent of the band (shared by both hemispheres).
    z_center = int(z_com + m['dz_center'] / sz)
    half_z = int(m['height_z'] / sz) // 2
    z0 = _clamp(z_center - half_z, 0, data.shape[2])
    z1 = _clamp(z_center + half_z, 0, data.shape[2])

    # Medial band on each side: |x| from x_inner to x_outer.
    bands = {}
    for side, sign in (('lh', -1), ('rh', 1)):
        xa = int(x_com + sign * m['x_inner'] / sx)
        xb = int(x_com + sign * m['x_outer'] / sx)
        lo, hi = (xa, xb) if xa <= xb else (xb, xa)
        bands[side] = (_clamp(lo, 0, data.shape[0]), _clamp(hi, 0, data.shape[0]))

    # Coronal slices from the hippocampal tail (dy_start) to the head (dy_end).
    dys = range(int(m['dy_start']), int(m['dy_end']) + 1)
    y_voxels = [_clamp(int(y_com + dy / sy), 0, data.shape[1] - 1) for dy in dys]

    per_side = {}
    for side, (x0, x1) in bands.items():
        vals = [_window_ratio(data[:, y, :], x0, x1, z0, z1, thresh) for y in y_voxels]
        per_side[side] = float(np.mean(vals)) if vals else 0.0

    y_mid = y_voxels[len(y_voxels) // 2]
    metadata = {
        'y_mta_vox': y_mid,                      # representative slice for display
        'y_mta_range': (y_voxels[0], y_voxels[-1]),
        'n_mta_slices': len(y_voxels),
        'lh_x': bands['lh'],
        'rh_x': bands['rh'],
        'z_mta': (z0, z1),
        'mta_thresh': float(thresh),
    }
    return per_side['lh'], per_side['rh'], metadata


def _lobar_boxes(region, data, x_com, y_com, spacing, cfg):
    """In-plane bounding boxes for a lobar window, as (x0, x1, y0, y1) tuples."""
    sx = abs(spacing[0])
    sy = spacing[1]
    c = cfg[region]

    if region == 'frontal':
        y0 = int(y_com + c['dy_offset'] / sy)
        y1 = int(y_com + (c['dy_offset'] + 40) / sy)
        x0 = int(x_com - 40 / sx)
        x1 = int(x_com + 40 / sx)
        boxes = [(x0, x1, y0, y1)]
    elif region == 'parietal':
        y0 = int(y_com + (c['dy_offset'] - 40) / sy)
        y1 = int(y_com + c['dy_offset'] / sy)
        x0 = int(x_com - 40 / sx)
        x1 = int(x_com + 40 / sx)
        boxes = [(x0, x1, y0, y1)]
    else:  # temporal: bilateral lateral boxes
        dx = c['dx_offset']
        y0 = int(y_com - 30 / sy)
        y1 = int(y_com + 30 / sy)
        lx0 = int(x_com - (dx + 25) / sx)
        lx1 = int(x_com - dx / sx)
        rx0 = int(x_com + dx / sx)
        rx1 = int(x_com + (dx + 25) / sx)
        boxes = [(lx0, lx1, y0, y1), (rx0, rx1, y0, y1)]

    clamped = []
    for (bx0, bx1, by0, by1) in boxes:
        clamped.append((_clamp(bx0, 0, data.shape[0] - 1), _clamp(bx1, 0, data.shape[0] - 1),
                        _clamp(by0, 0, data.shape[1] - 1), _clamp(by1, 0, data.shape[1] - 1)))
    return clamped


def extract_lobar_ratio(data, x_com, y_com, z_com, spacing, i98, cfg, region):
    """CSF fraction for one lobar window, averaged over adjacent axial slices."""
    c = cfg[region]
    thresh = c['tf'] * i98
    sz = spacing[2]

    z_center = _clamp(int(z_com + c['dz_offset'] / sz), 0, data.shape[2] - 1)
    # n_slices is a slice count (see config notes); 11 slices ~ +/- 5 mm at
    # the 1.0-1.05 mm slice spacing of the analysed data.
    n_half = int(c.get('n_slices', 1)) // 2

    boxes = _lobar_boxes(region, data, x_com, y_com, spacing, cfg)

    ratios = []
    for dz in range(-n_half, n_half + 1):
        z = _clamp(z_center + dz, 0, data.shape[2] - 1)
        csf = 0
        total = 0
        for (x0, x1, y0, y1) in boxes:
            roi = data[x0:x1, y0:y1, z]
            if roi.size > 0:
                csf += int((roi < thresh).sum())
                total += roi.size
        ratios.append(csf / total if total > 0 else 0.0)

    return float(np.mean(ratios)), z_center, boxes, 2 * n_half + 1


def extract_features(data, x_com, y_com, z_com, spacing, i98, cfg):
    """Extract the five regional CSF fractions and the window metadata.

    Returns:
        ratios: dict of the five transparent CSF-fraction features
        metadata: voxel bounds and thresholds for visualisation and QC
    """
    lt, rt, metadata = extract_medial_temporal_ratios(
        data, x_com, y_com, z_com, spacing, i98, cfg)

    ratio_front, z_f, f_boxes, n_f = extract_lobar_ratio(
        data, x_com, y_com, z_com, spacing, i98, cfg, 'frontal')
    ratio_parietal, z_p, p_boxes, n_p = extract_lobar_ratio(
        data, x_com, y_com, z_com, spacing, i98, cfg, 'parietal')
    ratio_temporal, z_t, t_boxes, n_t = extract_lobar_ratio(
        data, x_com, y_com, z_com, spacing, i98, cfg, 'temporal')

    ratios = {
        'Auto_Lt_Ratio': lt,
        'Auto_Rt_Ratio': rt,
        'Auto_Front_Ratio': ratio_front,
        'Auto_Parietal_Ratio': ratio_parietal,
        'Auto_Temporal_Ratio': ratio_temporal,
    }

    fx0, fx1, fy0, fy1 = f_boxes[0]
    px0, px1, py0, py1 = p_boxes[0]
    (tlx0, tlx1, ty0, ty1), (trx0, trx1, _, _) = t_boxes

    metadata.update({
        'z_front_vox': z_f, 'f_x': (fx0, fx1), 'f_y': (fy0, fy1), 'n_front_slices': n_f,
        'z_parietal_vox': z_p, 'p_x': (px0, px1), 'p_y': (py0, py1), 'n_parietal_slices': n_p,
        'z_temporal_vox': z_t, 't_x_l': (tlx0, tlx1), 't_x_r': (trx0, trx1),
        't_y': (ty0, ty1), 'n_temporal_slices': n_t,
        'front_thresh': float(cfg['frontal']['tf'] * i98),
        'parietal_thresh': float(cfg['parietal']['tf'] * i98),
        'temporal_thresh': float(cfg['temporal']['tf'] * i98),
    })

    return ratios, metadata
