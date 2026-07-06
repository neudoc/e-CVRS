import numpy as np

def calculate_com(data, i98):
    """
    Calculates the brain Center of Mass (COM) after stripping neck voxels.
    Returns:
        x_com, y_com, z_com, qc_flag, qc_message
    """
    # 1. Generate head mask
    brain_mask = data > (0.12 * i98)
    
    # 2. Find z_top (top of the brain)
    z_indices = np.argwhere(brain_mask)
    if z_indices.size == 0:
        return 0, 0, 0, False, "Empty brain mask"
        
    z_top = z_indices[:, 2].max()
    
    # 3. Strip neck: set voxels below z_top - 130mm to 0
    # Assuming 1mm axial slice spacing, z_top - 130 is 130 slices below the top.
    # In raw scans, z_min is 0.
    z_limit = int(z_top - 130)
    head_mask = brain_mask.copy()
    if z_limit > 0:
        head_mask[:, :, :z_limit] = False
        
    # 4. Calculate COM
    head_coords = np.argwhere(head_mask)
    if head_coords.size == 0:
        return 0, 0, 0, False, "Head mask empty after neck stripping"
        
    x_com, y_com, z_com = head_coords.mean(axis=0)
    
    # QC Checks
    qc_flag = True
    qc_message = "Pass"
    
    # If COM is too close to the borders, flag it
    if x_com < 10 or x_com > data.shape[0] - 10:
        qc_flag = False
        qc_message = "COM X is near borders"
    elif y_com < 20 or y_com > data.shape[1] - 20:
        qc_flag = False
        qc_message = "COM Y is near borders"
    elif z_com < 20 or z_com > data.shape[2] - 20:
        qc_flag = False
        qc_message = "COM Z is near borders"
        
    return x_com, y_com, z_com, qc_flag, qc_message
