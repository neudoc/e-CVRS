import numpy as np

def extract_features(data, x_com, y_com, z_com, spacing, i98, cfg):
    """
    Extracts the local CSF ratios and ROI coordinates for the e-CVRS subscales.
    Returns:
        ratios: dict of float ratios
        metadata: dict of coordinate slices and bounds for visualization
    """
    # 1. Hippocampus MTA Left/Right
    # Coronal slice y_mta
    dy_mta = cfg['mta']['dy_offset']
    dz_mta = cfg['mta']['dz_offset']
    dx_mta = cfg['mta']['dx_offset']
    tf_mta = cfg['mta']['tf']
    
    y_mta_vox = int(y_com + dy_mta / spacing[1])
    z_mta_vox = int(z_com + dz_mta / spacing[2])
    dx_l_vox = int(-dx_mta / abs(spacing[0]))
    dx_r_vox = int(dx_mta / abs(spacing[0]))
    
    w_x_vox = int(35 / abs(spacing[0]))
    w_z_vox = int(25 / spacing[2])
    
    lh_x_start = max(0, int(x_com + dx_l_vox - w_x_vox//2))
    lh_x_end = min(data.shape[0], int(x_com + dx_l_vox + w_x_vox//2))
    rh_x_start = max(0, int(x_com + dx_r_vox - w_x_vox//2))
    rh_x_end = min(data.shape[0], int(x_com + dx_r_vox + w_x_vox//2))
    
    z_mta_start = max(0, int(z_mta_vox - w_z_vox//2))
    z_mta_end = min(data.shape[2], int(z_mta_vox + w_z_vox//2))
    
    csf_thresh_mta = tf_mta * i98
    
    # 5-slice average for MTA
    l_mta_ratios = []
    r_mta_ratios = []
    for y in range(y_mta_vox - 2, y_mta_vox + 3):
        y_idx = max(0, min(data.shape[1]-1, y))
        sd = data[:, y_idx, :]
        l_roi = sd[lh_x_start:lh_x_end, z_mta_start:z_mta_end]
        r_roi = sd[rh_x_start:rh_x_end, z_mta_start:z_mta_end]
        l_mta_ratios.append((l_roi < csf_thresh_mta).sum() / l_roi.size if l_roi.size > 0 else 0)
        r_mta_ratios.append((r_roi < csf_thresh_mta).sum() / r_roi.size if r_roi.size > 0 else 0)
        
    ratio_hippo_lt = np.mean(l_mta_ratios)
    ratio_hippo_rt = np.mean(r_mta_ratios)
    
    # 2. Frontal Lobe Atrophy
    dz_front = cfg['frontal']['dz_offset']
    dy_front = cfg['frontal']['dy_offset']
    tf_front = cfg['frontal']['tf']
    
    z_front_vox = int(z_com + dz_front / spacing[2])
    f_y_start = int(y_com + dy_front / spacing[1])
    f_y_end = int(y_com + (dy_front + 40) / spacing[1])
    f_x_start = int(x_com - 40 / abs(spacing[0]))
    f_x_end = int(x_com + 40 / abs(spacing[0]))
    
    z_front_vox = max(0, min(data.shape[2]-1, z_front_vox))
    f_y_start = max(0, min(data.shape[1]-1, f_y_start))
    f_y_end = max(0, min(data.shape[1]-1, f_y_end))
    f_x_start = max(0, min(data.shape[0]-1, f_x_start))
    f_x_end = max(0, min(data.shape[0]-1, f_x_end))
    
    csf_thresh_front = tf_front * i98
    f_roi = data[f_x_start:f_x_end, f_y_start:f_y_end, z_front_vox]
    ratio_front = (f_roi < csf_thresh_front).sum() / f_roi.size if f_roi.size > 0 else 0
    
    # 3. Parietal Lobe Atrophy
    dz_parietal = cfg['parietal']['dz_offset']
    dy_parietal = cfg['parietal']['dy_offset']
    tf_parietal = cfg['parietal']['tf']
    
    z_parietal_vox = int(z_com + dz_parietal / spacing[2])
    p_y_start = int(y_com + (dy_parietal - 40) / spacing[1])
    p_y_end = int(y_com + dy_parietal / spacing[1])
    p_x_start = int(x_com - 40 / abs(spacing[0]))
    p_x_end = int(x_com + 40 / abs(spacing[0]))
    
    z_parietal_vox = max(0, min(data.shape[2]-1, z_parietal_vox))
    p_y_start = max(0, min(data.shape[1]-1, p_y_start))
    p_y_end = max(0, min(data.shape[1]-1, p_y_end))
    p_x_start = max(0, min(data.shape[0]-1, p_x_start))
    p_x_end = max(0, min(data.shape[0]-1, p_x_end))
    
    csf_thresh_parietal = tf_parietal * i98
    p_roi = data[p_x_start:p_x_end, p_y_start:p_y_end, z_parietal_vox]
    ratio_parietal = (p_roi < csf_thresh_parietal).sum() / p_roi.size if p_roi.size > 0 else 0
    
    # 4. Temporal Lobe Atrophy
    dz_temporal = cfg['temporal']['dz_offset']
    dx_temporal = cfg['temporal']['dx_offset']
    tf_temporal = cfg['temporal']['tf']
    
    z_temporal_vox = int(z_com + dz_temporal / spacing[2])
    t_y_start = int(y_com - 30 / spacing[1])
    t_y_end = int(y_com + 30 / spacing[1])
    t_x_l_start = int(x_com - (dx_temporal + 25) / abs(spacing[0]))
    t_x_l_end = int(x_com - dx_temporal / abs(spacing[0]))
    t_x_r_start = int(x_com + dx_temporal / abs(spacing[0]))
    t_x_r_end = int(x_com + (dx_temporal + 25) / abs(spacing[0]))
    
    z_temporal_vox = max(0, min(data.shape[2]-1, z_temporal_vox))
    t_y_start = max(0, min(data.shape[1]-1, t_y_start))
    t_y_end = max(0, min(data.shape[1]-1, t_y_end))
    t_x_l_start = max(0, min(data.shape[0]-1, t_x_l_start))
    t_x_l_end = max(0, min(data.shape[0]-1, t_x_l_end))
    t_x_r_start = max(0, min(data.shape[0]-1, t_x_r_start))
    t_x_r_end = max(0, min(data.shape[0]-1, t_x_r_end))
    
    csf_thresh_temporal = tf_temporal * i98
    t_roi_l = data[t_x_l_start:t_x_l_end, t_y_start:t_y_end, z_temporal_vox]
    t_roi_r = data[t_x_r_start:t_x_r_end, t_y_start:t_y_end, z_temporal_vox]
    
    csf_sum = 0
    total_size = 0
    if t_roi_l.size > 0:
        csf_sum += (t_roi_l < csf_thresh_temporal).sum()
        total_size += t_roi_l.size
    if t_roi_r.size > 0:
        csf_sum += (t_roi_r < csf_thresh_temporal).sum()
        total_size += t_roi_r.size
    ratio_temporal = csf_sum / total_size if total_size > 0 else 0
    
    # 5-slice averages of cortical regions to stabilize values (Optional or for future use)
    ratios = {
        'Auto_Lt_Ratio': ratio_hippo_lt,
        'Auto_Rt_Ratio': ratio_hippo_rt,
        'Auto_Front_Ratio': ratio_front,
        'Auto_Parietal_Ratio': ratio_parietal,
        'Auto_Temporal_Ratio': ratio_temporal
    }
    
    metadata = {
        'y_mta_vox': y_mta_vox,
        'lh_x': (lh_x_start, lh_x_end),
        'rh_x': (rh_x_start, rh_x_end),
        'z_mta': (z_mta_start, z_mta_end),
        
        'z_front_vox': z_front_vox,
        'f_x': (f_x_start, f_x_end),
        'f_y': (f_y_start, f_y_end),
        
        'z_parietal_vox': z_parietal_vox,
        'p_x': (p_x_start, p_x_end),
        'p_y': (p_y_start, p_y_end),
        
        'z_temporal_vox': z_temporal_vox,
        't_x_l': (t_x_l_start, t_x_l_end),
        't_x_r': (t_x_r_start, t_x_r_end),
        't_y': (t_y_start, t_y_end)
    }
    
    return ratios, metadata
