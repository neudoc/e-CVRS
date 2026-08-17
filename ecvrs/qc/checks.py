def run_qc_checks(shape, x_com, y_com, z_com, r_meta, i98):
    """
    Runs structural and alignment QC checks on a subject scan.
    Returns:
        is_pass: bool
        issues: list of str reasons for failure
    """
    is_pass = True
    issues = []
    
    # 1. Intensity Check
    if i98 <= 0:
        is_pass = False
        issues.append("98th percentile intensity is zero or negative")
        
    # 2. COM Check
    if x_com == 0 and y_com == 0 and z_com == 0:
        is_pass = False
        issues.append("Center of Mass calculation failed")
        
    # 3. ROI Out of Bounds Check
    # Check coronal slice boundaries
    y_mta = r_meta.get('y_mta_vox', 0)
    if y_mta < 0 or y_mta >= shape[1]:
        is_pass = False
        issues.append(f"MTA coronal slice y={y_mta} is out of bounds (0-{shape[1]-1})")
        
    # Check Frontal box boundaries
    f_x = r_meta.get('f_x', (0, 0))
    f_y = r_meta.get('f_y', (0, 0))
    z_f = r_meta.get('z_front_vox', 0)
    if f_x[0] < 0 or f_x[1] > shape[0] or f_y[0] < 0 or f_y[1] > shape[1] or z_f < 0 or z_f >= shape[2]:
        is_pass = False
        issues.append("Frontal ROI bounds out of scan dimensions")
        
    # Check Parietal box boundaries
    p_x = r_meta.get('p_x', (0, 0))
    p_y = r_meta.get('p_y', (0, 0))
    z_p = r_meta.get('z_parietal_vox', 0)
    if p_x[0] < 0 or p_x[1] > shape[0] or p_y[0] < 0 or p_y[1] > shape[1] or z_p < 0 or z_p >= shape[2]:
        is_pass = False
        issues.append("Parietal ROI bounds out of scan dimensions")
        
    # Check Temporal box boundaries
    t_xl = r_meta.get('t_x_l', (0, 0))
    t_xr = r_meta.get('t_x_r', (0, 0))
    t_y = r_meta.get('t_y', (0, 0))
    z_t = r_meta.get('z_temporal_vox', 0)
    if t_xl[0] < 0 or t_xl[1] > shape[0] or t_xr[0] < 0 or t_xr[1] > shape[0] or t_y[0] < 0 or t_y[1] > shape[1] or z_t < 0 or z_t >= shape[2]:
        is_pass = False
        issues.append("Temporal ROI bounds out of scan dimensions")
        
    return is_pass, issues
