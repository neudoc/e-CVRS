import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_mri_overlays(data, x_com, y_com, z_com, spacing, i98, r_meta, scores, ratios, thresholds, output_dir, rid):
    """
    Generates and saves explanatory ROI visual overlays for a single subject.
    Saves:
        <output_dir>/overlays/<rid>_summary.png - A multi-panel panel of all ROIs
    """
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
    
    # 1. Coronal Slice for MTA (Hippocampus)
    y_mta = r_meta['y_mta_vox']
    slice_mta = data[:, y_mta, :]
    
    # Define thresholds
    csf_thresh = 0.25 * i98 # MTA factor
    
    # Coordinates
    lh_x = r_meta['lh_x']
    rh_x = r_meta['rh_x']
    z_mta = r_meta['z_mta']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"e-CVRS Explanatory Visual Overlay - Subject RID: {rid}\nTotal Sum Score: {scores['e_Atrophy_Sum']}/17", fontsize=16, fontweight='bold')
    
    # Panel 1: Medial Temporal Atrophy (MTA)
    ax = axes[0, 0]
    ax.imshow(slice_mta.T, cmap='gray', origin='lower')
    
    # Draw left and right boxes
    rect_l = patches.Rectangle((lh_x[0], z_mta[0]), lh_x[1]-lh_x[0], z_mta[1]-z_mta[0], linewidth=1.5, edgecolor='cyan', facecolor='none', label='Left Hippo ROI')
    rect_r = patches.Rectangle((rh_x[0], z_mta[0]), rh_x[1]-rh_x[0], z_mta[1]-z_mta[0], linewidth=1.5, edgecolor='magenta', facecolor='none', label='Right Hippo ROI')
    ax.add_patch(rect_l)
    ax.add_patch(rect_r)
    
    # Highlight CSF inside ROIs
    # Create mask for Left ROI
    l_roi_mask = np.zeros_like(slice_mta, dtype=bool)
    l_roi_mask[lh_x[0]:lh_x[1], z_mta[0]:z_mta[1]] = slice_mta[lh_x[0]:lh_x[1], z_mta[0]:z_mta[1]] < csf_thresh
    
    r_roi_mask = np.zeros_like(slice_mta, dtype=bool)
    r_roi_mask[rh_x[0]:rh_x[1], z_mta[0]:z_mta[1]] = slice_mta[rh_x[0]:rh_x[1], z_mta[0]:z_mta[1]] < csf_thresh
    
    # Display CSF as overlay
    ax.imshow(np.where(l_roi_mask.T, 1, np.nan), cmap='Blues_r', origin='lower', alpha=0.6)
    ax.imshow(np.where(r_roi_mask.T, 1, np.nan), cmap='RdPu_r', origin='lower', alpha=0.6)
    
    ax.set_title(f"MTA Coronal (y={y_mta})\nLt Ratio: {ratios['Auto_Lt_Ratio']:.3f} (Score {scores['e_Hippo_Lt']})\nRt Ratio: {ratios['Auto_Rt_Ratio']:.3f} (Score {scores['e_Hippo_Rt']})")
    ax.set_xlim(x_com - 60, x_com + 60)
    ax.set_ylim(z_com - 40, z_com + 20)
    ax.axis('off')
    
    # Panel 2: Frontal Lobe Atrophy
    ax = axes[0, 1]
    z_f = r_meta['z_front_vox']
    slice_f = data[:, :, z_f]
    f_x = r_meta['f_x']
    f_y = r_meta['f_y']
    
    ax.imshow(slice_f.T, cmap='gray', origin='lower')
    rect_f = patches.Rectangle((f_x[0], f_y[0]), f_x[1]-f_x[0], f_y[1]-f_y[0], linewidth=1.5, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect_f)
    
    # CSF mask
    f_mask = np.zeros_like(slice_f, dtype=bool)
    f_mask[f_x[0]:f_x[1], f_y[0]:f_y[1]] = slice_f[f_x[0]:f_x[1], f_y[0]:f_y[1]] < (0.30 * i98)
    ax.imshow(np.where(f_mask.T, 1, np.nan), cmap='Wistia_r', origin='lower', alpha=0.6)
    
    ax.set_title(f"Frontal Lobe Axial (z={z_f})\nRatio: {ratios['Auto_Front_Ratio']:.3f} (Score {scores['e_Front']})")
    ax.set_xlim(x_com - 60, x_com + 60)
    ax.set_ylim(y_com - 20, y_com + 70)
    ax.axis('off')
    
    # Panel 3: Parietal Lobe Atrophy
    ax = axes[1, 0]
    z_p = r_meta['z_parietal_vox']
    slice_p = data[:, :, z_p]
    p_x = r_meta['p_x']
    p_y = r_meta['p_y']
    
    ax.imshow(slice_p.T, cmap='gray', origin='lower')
    rect_p = patches.Rectangle((p_x[0], p_y[0]), p_x[1]-p_x[0], p_y[1]-p_y[0], linewidth=1.5, edgecolor='orange', facecolor='none')
    ax.add_patch(rect_p)
    
    p_mask = np.zeros_like(slice_p, dtype=bool)
    p_mask[p_x[0]:p_x[1], p_y[0]:p_y[1]] = slice_p[p_x[0]:p_x[1], p_y[0]:p_y[1]] < (0.22 * i98)
    ax.imshow(np.where(p_mask.T, 1, np.nan), cmap='Oranges_r', origin='lower', alpha=0.6)
    
    ax.set_title(f"Parietal Lobe Axial (z={z_p})\nRatio: {ratios['Auto_Parietal_Ratio']:.3f} (Score {scores['e_Parietal']})")
    ax.set_xlim(x_com - 60, x_com + 60)
    ax.set_ylim(y_com - 70, y_com + 30)
    ax.axis('off')
    
    # Panel 4: Temporal Lobe Atrophy
    ax = axes[1, 1]
    z_t = r_meta['z_temporal_vox']
    slice_t = data[:, :, z_t]
    t_xl = r_meta['t_x_l']
    t_xr = r_meta['t_x_r']
    t_y = r_meta['t_y']
    
    ax.imshow(slice_t.T, cmap='gray', origin='lower')
    rect_tl = patches.Rectangle((t_xl[0], t_y[0]), t_xl[1]-t_xl[0], t_y[1]-t_y[0], linewidth=1.5, edgecolor='green', facecolor='none')
    rect_tr = patches.Rectangle((t_xr[0], t_y[0]), t_xr[1]-t_xr[0], t_y[1]-t_y[0], linewidth=1.5, edgecolor='green', facecolor='none')
    ax.add_patch(rect_tl)
    ax.add_patch(rect_tr)
    
    t_mask = np.zeros_like(slice_t, dtype=bool)
    t_mask[t_xl[0]:t_xl[1], t_y[0]:t_y[1]] = slice_t[t_xl[0]:t_xl[1], t_y[0]:t_y[1]] < (0.30 * i98)
    t_mask[t_xr[0]:t_xr[1], t_y[0]:t_y[1]] = slice_t[t_xr[0]:t_xr[1], t_y[0]:t_y[1]] < (0.30 * i98)
    ax.imshow(np.where(t_mask.T, 1, np.nan), cmap='Greens_r', origin='lower', alpha=0.6)
    
    ax.set_title(f"Temporal Lobe Axial (z={z_t})\nRatio: {ratios['Auto_Temporal_Ratio']:.3f} (Score {scores['e_Temporal']})")
    ax.set_xlim(x_com - 75, x_com + 75)
    ax.set_ylim(y_com - 50, y_com + 50)
    ax.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'overlays', f"{rid}_summary.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
