"""Explanatory overlay that shows exactly what produced each e-CVRS grade.

For one scan the figure shows
  * the medial temporal windows on three coronal slices (hippocampal tail, body, head)
    of the 31-slice series that is averaged,
  * the frontal, parietal and temporal windows on the centre slice of the 11-slice series,
  * the CSF masks drawn with the SAME thresholds (configured fraction of I98) that the
    features use,
  * a panel listing the 11 feature values and, per region, the continuous model score,
    the locked model's cut-offs, and the resulting grade.
Grades come from the locked calibration model (ecvrs/models/locked_model_adni_mci.json),
so the overlay always matches the reported e-CVRS grades.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ecvrs.features.panel import build_panel, FEATURE_NAMES
from ecvrs.calibration.quadratic_ridge import score_quadratic_ridge
from ecvrs.calibration.ordinal import predict_kappa_optimized_mapping

REGIONS = [('Hippo_Lt', 'MTA left'), ('Hippo_Rt', 'MTA right'), ('Front', 'Frontal'),
           ('Parietal', 'Parietal'), ('Temporal', 'Temporal')]


def apply_locked_model(features, model):
    """features: dict of raw ratios/volumes for one scan -> {region: (score, cutoffs, grade)}."""
    X = build_panel(features)
    out = {}
    for region, _ in REGIONS:
        m = model['regions'][region]
        mm = {k: np.asarray(v) if isinstance(v, list) else v for k, v in m.items()}
        s = float(score_quadratic_ridge(X, mm)[0])
        g = int(predict_kappa_optimized_mapping(np.array([s]), np.asarray(m['thresholds']), m['num_classes'])[0])
        out[region] = (s, list(m['thresholds']), g)
    return out, X[0]


def _mask(ax, plane, boxes, thresh):
    for (x0, x1, y0, y1, col) in boxes:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=col, facecolor='none', linewidth=1.6))
        m = np.full(plane.shape, np.nan)
        m[x0:x1, y0:y1] = np.where(plane[x0:x1, y0:y1] < thresh, 1.0, np.nan)
        ax.imshow(m.T, cmap='autumn', origin='lower', alpha=0.75, interpolation='nearest')


def render_explanation(data, spacing, com, i98, features, meta, cfg, model, rid,
                       output_path, expert=None):
    x_com, y_com, z_com = com
    sx, sy, sz = abs(spacing[0]), abs(spacing[1]), abs(spacing[2])
    grades, xvec = apply_locked_model(features, model)
    total = sum(g for _, _, g in grades.values())

    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15])
    lh, rh, zm = meta['lh_x'], meta['rh_x'], meta['z_mta']
    t_mta = cfg['mta']['tf'] * i98
    for k, (lab, dy) in enumerate([('tail', cfg['mta']['dy_start']), ('body', (cfg['mta']['dy_start'] + cfg['mta']['dy_end']) / 2),
                                   ('head', cfg['mta']['dy_end'])]):
        ax = fig.add_subplot(gs[0, k])
        y = int(np.clip(y_com + dy / sy, 0, data.shape[1] - 1)); plane = data[:, y, :]
        ax.imshow(plane.T, cmap='gray', origin='lower', vmin=0, vmax=1.1 * i98)
        _mask(ax, plane, [(lh[0], lh[1], zm[0], zm[1], 'cyan'), (rh[0], rh[1], zm[0], zm[1], 'magenta')], t_mta)
        ax.axhline(z_com, color='orange', ls=':', lw=1)
        ax.set_xlim(x_com - 70 / sx, x_com + 70 / sx); ax.set_ylim(z_com - 55 / sz, z_com + 35 / sz); ax.axis('off')
        ax.set_title(f'MTA windows, hippocampal {lab} (y = COM {dy:+.0f} mm)\n'
                     f'one of {meta.get("n_mta_slices", 31)} averaged coronal slices', fontsize=9)
    for k, (key, zkey, xk, yk, lab) in enumerate([('frontal', 'z_front_vox', ['f_x'], 'f_y', 'Frontal'),
                                                  ('parietal', 'z_parietal_vox', ['p_x'], 'p_y', 'Parietal'),
                                                  ('temporal', 'z_temporal_vox', ['t_x_l', 't_x_r'], 't_y', 'Temporal')]):
        ax = fig.add_subplot(gs[1, k]); z = meta[zkey]; plane = data[:, :, z]
        ax.imshow(plane.T, cmap='gray', origin='lower', vmin=0, vmax=1.1 * i98)
        y0, y1 = meta[yk]
        _mask(ax, plane, [(meta[x][0], meta[x][1], y0, y1, 'yellow') for x in xk], cfg[key]['tf'] * i98)
        ax.set_xlim(x_com - 85 / sx, x_com + 85 / sx); ax.set_ylim(y_com - 95 / sy, y_com + 95 / sy); ax.axis('off')
        ax.set_title(f'{lab} window, centre of {cfg[key].get("n_slices", 1)} averaged axial slices\n'
                     f'CSF threshold {cfg[key]["tf"]:.2f} x I98', fontsize=9)
    ax = fig.add_subplot(gs[:, 3]); ax.axis('off')
    lines = ['Feature panel (model input)']
    lines += [f'  {n:<22s}{v:>9.4f}' for n, v in zip(FEATURE_NAMES, xvec)]
    lines += ['', 'Locked model: score -> cut-offs -> grade']
    for region, name in REGIONS:
        s, cuts, g = grades[region]
        ex = '' if expert is None else f'  (expert {int(expert[region])})'
        lines.append(f'  {name:<10s} score {s:5.2f}  grade {g}{ex}')
        lines.append(f'     cut-offs ' + ', '.join(f'{c:.2f}' for c in cuts))
    ex_tot = '' if expert is None else f'  (expert {int(sum(expert[r] for r, _ in REGIONS))})'
    lines += ['', f'  TOTAL atrophy score {total} / 17{ex_tot}', '',
              'Masks: voxels below the configured', 'fraction of I98 inside each window.',
              'Cyan / magenta: left / right MTA windows.', 'Dotted line: centre-of-mass plane.']
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=9.2, transform=ax.transAxes)
    fig.suptitle(f'e-CVRS explanatory overlay - RID {rid}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150); plt.close(fig)
    return grades, total
