import numpy as np

def extract_proxy_volumes(data, x_com, y_com, z_com, spacing, i98):
    """
    Calculates 3D box-based proxy volumes (exploratory proxies).
    Returns:
        volumes: dict of float volumes in mm^3
    """
    vox_vol = abs(spacing[0] * spacing[1] * spacing[2])
    
    # Brain mask threshold
    brain_mask = data > (0.12 * i98)
    z_indices = np.argwhere(brain_mask)
    if z_indices.size == 0:
        return {}
    z_top = z_indices[:, 2].max()
    
    z_limit = int(z_top - 130)
    head_mask = brain_mask.copy()
    if z_limit > 0:
        head_mask[:, :, :z_limit] = False
        
    vol_tbv = head_mask.sum() * vox_vol
    
    # Offsets and boxes
    dy_mta_vox = int(-12 / spacing[1])
    dz_mta_vox = int(-12 / spacing[2])
    dx_l_vox = int(-28 / abs(spacing[0]))
    dx_r_vox = int(28 / abs(spacing[0]))
    
    w_x_vox = int(35 / abs(spacing[0]))
    w_y_vox = int(20 / spacing[1])
    w_z_vox = int(25 / spacing[2])
    
    lh_x_start = max(0, int(x_com + dx_l_vox - w_x_vox//2))
    lh_x_end = min(data.shape[0], int(x_com + dx_l_vox + w_x_vox//2))
    lh_y_start = max(0, int(y_com + dy_mta_vox - w_y_vox//2))
    lh_y_end = min(data.shape[1], int(y_com + dy_mta_vox + w_y_vox//2))
    lh_z_start = max(0, int(z_com + dz_mta_vox - w_z_vox//2))
    lh_z_end = min(data.shape[2], int(z_com + dz_mta_vox + w_z_vox//2))
    
    rh_x_start = max(0, int(x_com + dx_r_vox - w_x_vox//2))
    rh_x_end = min(data.shape[0], int(x_com + dx_r_vox + w_x_vox//2))
    
    # Central Ventricle box
    v_x_start = max(0, int(x_com - 25 / abs(spacing[0])))
    v_x_end = min(data.shape[0], int(x_com + 25 / abs(spacing[0])))
    v_y_start = max(0, int(y_com - 30 / spacing[1]))
    v_y_end = min(data.shape[1], int(y_com + 30 / spacing[1]))
    v_z_start = max(0, int(z_com - 20 / spacing[2]))
    v_z_end = min(data.shape[2], int(z_com + 20 / spacing[2]))
    
    csf_thresh_25 = 0.25 * i98
    tissue_thresh_upper = 0.95 * i98
    
    # Left Hippo tissue volume
    lh_roi_3d = data[lh_x_start:lh_x_end, lh_y_start:lh_y_end, lh_z_start:lh_z_end]
    vol_hippo_lt = ((lh_roi_3d >= csf_thresh_25) & (lh_roi_3d < tissue_thresh_upper)).sum() * vox_vol
    
    # Right Hippo tissue volume
    rh_roi_3d = data[rh_x_start:rh_x_end, lh_y_start:lh_y_end, lh_z_start:lh_z_end]
    vol_hippo_rt = ((rh_roi_3d >= csf_thresh_25) & (rh_roi_3d < tissue_thresh_upper)).sum() * vox_vol
    
    # Ventricle CSF volume
    v_roi_3d = data[v_x_start:v_x_end, v_y_start:v_y_end, v_z_start:v_z_end]
    vol_ventricle = (v_roi_3d < csf_thresh_25).sum() * vox_vol
    
    # Cortical Lobe tissue volume proxies
    # Frontal (above COM, anterior)
    f_z_start = max(0, int(z_com + 20 / spacing[2]))
    f_z_end = min(data.shape[2], int(z_com + 60 / spacing[2]))
    f_y_start = max(0, int(y_com + 10 / spacing[1]))
    f_y_end = min(data.shape[1], int(y_com + 60 / spacing[1]))
    f_x_start = max(0, int(x_com - 40 / abs(spacing[0])))
    f_x_end = min(data.shape[0], int(x_com + 40 / abs(spacing[0])))
    f_roi_3d = data[f_x_start:f_x_end, f_y_start:f_y_end, f_z_start:f_z_end]
    vol_front = ((f_roi_3d >= csf_thresh_25) & (f_roi_3d < tissue_thresh_upper)).sum() * vox_vol
    
    # Parietal (above COM, posterior)
    p_z_start = max(0, int(z_com + 25 / spacing[2]))
    p_z_end = min(data.shape[2], int(z_com + 65 / spacing[2]))
    p_y_start = max(0, int(y_com - 60 / spacing[1]))
    p_y_end = min(data.shape[1], int(y_com - 10 / spacing[1]))
    p_x_start = max(0, int(x_com - 40 / abs(spacing[0])))
    p_x_end = min(data.shape[0], int(x_com + 40 / abs(spacing[0])))
    p_roi_3d = data[p_x_start:p_x_end, p_y_start:p_y_end, p_z_start:p_z_end]
    vol_parietal = ((p_roi_3d >= csf_thresh_25) & (p_roi_3d < tissue_thresh_upper)).sum() * vox_vol
    
    # Temporal (below COM, lateral)
    t_z_start = max(0, int(z_com - 20 / spacing[2]))
    t_z_end = min(data.shape[2], int(z_com + 20 / spacing[2]))
    t_y_start = max(0, int(y_com - 30 / spacing[1]))
    t_y_end = min(data.shape[1], int(y_com + 30 / spacing[1]))
    t_x_l_start = max(0, int(x_com - 60 / abs(spacing[0])))
    t_x_l_end = min(data.shape[0], int(x_com - 35 / abs(spacing[0])))
    t_x_r_start = max(0, int(x_com + 35 / abs(spacing[0])))
    t_x_r_end = min(data.shape[0], int(x_com + 60 / abs(spacing[0])))
    
    t_roi_l_3d = data[t_x_l_start:t_x_l_end, t_y_start:t_y_end, t_z_start:t_z_end]
    t_roi_r_3d = data[t_x_r_start:t_x_r_end, t_y_start:t_y_end, t_z_start:t_z_end]
    
    vol_temporal_l = ((t_roi_l_3d >= csf_thresh_25) & (t_roi_l_3d < tissue_thresh_upper)).sum() * vox_vol
    vol_temporal_r = ((t_roi_r_3d >= csf_thresh_25) & (t_roi_r_3d < tissue_thresh_upper)).sum() * vox_vol
    vol_temporal = vol_temporal_l + vol_temporal_r
    
    return {
        'Vol_Hippo_Lt': vol_hippo_lt,
        'Vol_Hippo_Rt': vol_hippo_rt,
        'Vol_Front': vol_front,
        'Vol_Parietal': vol_parietal,
        'Vol_Temporal': vol_temporal,
        'Vol_Ventricle': vol_ventricle,
        'Vol_TBV': vol_tbv
    }
