import nibabel as nib
import numpy as np
import os
import pandas as pd
import re
import sys
import argparse

def process_subject(filepath, rid):
    try:
        img = nib.load(filepath)
        # Load header info and spacing
        spacing = np.diag(img.affine)[:3]
        vox_vol = abs(spacing[0] * spacing[1] * spacing[2]) # voxel volume in mm^3
        
        # Load image data
        data = img.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
            
        # Optimization: calculate percentile on downsampled data for speed
        i98 = np.percentile(data[::2, ::2, ::2], 98)
        
        # Brain mask threshold
        brain_mask = data > (0.12 * i98)
        
        # Find z_top (top of the brain)
        z_indices = np.argwhere(brain_mask)
        if z_indices.size == 0:
            return None
        z_top = z_indices[:, 2].max()
        
        # Remove neck: only keep voxels where z > z_top - 130
        z_limit = int(z_top - 130)
        head_mask = brain_mask.copy()
        head_mask[:, :, :z_limit] = False
        
        head_coords = np.argwhere(head_mask)
        if head_coords.size == 0:
            return None
        
        # Center of Mass (COM)
        x_com, y_com, z_com = head_coords.mean(axis=0)
        
        # Voxel calculations for 3D Volume Proxies
        # Total Brain Volume (TBV) in mm^3
        vol_tbv = head_mask.sum() * vox_vol
        
        # 1. Hippocampus ROI 3D Bounding Boxes (for volumetry proxy)
        # We define a 3D box of size 35mm (X) x 20mm (Y) x 25mm (Z) centered at the hippo COM offsets
        dy_mta_vox = int(-12 / spacing[1])
        dz_mta_vox = int(-12 / spacing[2])
        dx_l_vox = int(-28 / abs(spacing[0]))
        dx_r_vox = int(28 / abs(spacing[0]))
        
        w_x_vox = int(35 / abs(spacing[0]))
        w_y_vox = int(20 / spacing[1])
        w_z_vox = int(25 / spacing[2])
        
        # Left Hippo 3D ROI bounds
        lh_x_start = max(0, int(x_com + dx_l_vox - w_x_vox//2))
        lh_x_end = min(data.shape[0], int(x_com + dx_l_vox + w_x_vox//2))
        lh_y_start = max(0, int(y_com + dy_mta_vox - w_y_vox//2))
        lh_y_end = min(data.shape[1], int(y_com + dy_mta_vox + w_y_vox//2))
        lh_z_start = max(0, int(z_com + dz_mta_vox - w_z_vox//2))
        lh_z_end = min(data.shape[2], int(z_com + dz_mta_vox + w_z_vox//2))
        
        # Right Hippo 3D ROI bounds
        rh_x_start = max(0, int(x_com + dx_r_vox - w_x_vox//2))
        rh_x_end = min(data.shape[0], int(x_com + dx_r_vox + w_x_vox//2))
        rh_y_start = lh_y_start
        rh_y_end = lh_y_end
        rh_z_start = lh_z_start
        rh_z_end = lh_z_end
        
        # 2. Ventricles 3D Bounding Box
        v_x_start = max(0, int(x_com - 25 / abs(spacing[0])))
        v_x_end = min(data.shape[0], int(x_com + 25 / abs(spacing[0])))
        v_y_start = max(0, int(y_com - 30 / spacing[1]))
        v_y_end = min(data.shape[1], int(y_com + 30 / spacing[1]))
        v_z_start = max(0, int(z_com - 20 / spacing[2]))
        v_z_end = min(data.shape[2], int(z_com + 20 / spacing[2]))
        
        # Thresholds
        csf_thresh_22 = 0.22 * i98
        csf_thresh_25 = 0.25 * i98
        csf_thresh_30 = 0.30 * i98
        tissue_thresh_upper = 0.95 * i98
        
        # Calculate 3D Volumetry Proxies (Brain tissue: GM + WM)
        # Left Hippo tissue volume
        lh_roi_3d = data[lh_x_start:lh_x_end, lh_y_start:lh_y_end, lh_z_start:lh_z_end]
        vol_hippo_lt = ((lh_roi_3d >= csf_thresh_25) & (lh_roi_3d < tissue_thresh_upper)).sum() * vox_vol
        
        # Right Hippo tissue volume
        rh_roi_3d = data[rh_x_start:rh_x_end, rh_y_start:rh_y_end, rh_z_start:rh_z_end]
        vol_hippo_rt = ((rh_roi_3d >= csf_thresh_25) & (rh_roi_3d < tissue_thresh_upper)).sum() * vox_vol
        
        # Ventricle CSF volume
        v_roi_3d = data[v_x_start:v_x_end, v_y_start:v_y_end, v_z_start:v_z_end]
        vol_ventricle = (v_roi_3d < csf_thresh_25).sum() * vox_vol
        
        # 3. 2D Visual Rating CSF Ratios (e-CVRS features)
        # A. Medial Temporal Atrophy (MTA) (Coronal slice, dynamic average of 5 slices)
        y_mta = int(y_com + dy_mta_vox)
        
        l_mta_ratios = []
        r_mta_ratios = []
        for y in range(y_mta - 2, y_mta + 3):
            sd = data[:, y, :]
            l_roi = sd[lh_x_start:lh_x_end, lh_z_start:lh_z_end]
            r_roi = sd[rh_x_start:rh_x_end, lh_z_start:lh_z_end]
            l_mta_ratios.append((l_roi < csf_thresh_25).sum() / l_roi.size)
            r_mta_ratios.append((r_roi < csf_thresh_25).sum() / r_roi.size)
            
        ratio_hippo_lt = np.mean(l_mta_ratios)
        ratio_hippo_rt = np.mean(r_mta_ratios)
        
        # B. Frontal Atrophy (Axial slice dz = 20, anterior y_offset = 10, tf = 0.3)
        z_front = int(z_com + 20 / spacing[2])
        f_y_start = int(y_com + 10 / spacing[1])
        f_y_end = int(y_com + 50 / spacing[1])
        f_x_start = int(x_com - 40 / abs(spacing[0]))
        f_x_end = int(x_com + 40 / abs(spacing[0]))
        
        f_roi = data[f_x_start:f_x_end, f_y_start:f_y_end, z_front]
        ratio_front = (f_roi < csf_thresh_30).sum() / f_roi.size if f_roi.size > 0 else 0
        vol_front = ((data[f_x_start:f_x_end, f_y_start:f_y_end, :] >= csf_thresh_30) & (data[f_x_start:f_x_end, f_y_start:f_y_end, :] < tissue_thresh_upper)).sum() * vox_vol
        
        # C. Parietal Atrophy (Axial slice dz = 35, posterior y_offset = -10, tf = 0.2)
        z_parietal = int(z_com + 35 / spacing[2])
        p_y_start = int(y_com - 50 / spacing[1])
        p_y_end = int(y_com - 10 / spacing[1])
        p_x_start = int(x_com - 40 / abs(spacing[0]))
        p_x_end = int(x_com + 40 / abs(spacing[0]))
        
        p_roi = data[p_x_start:p_x_end, p_y_start:p_y_end, z_parietal]
        ratio_parietal = (p_roi < csf_thresh_22).sum() / p_roi.size if p_roi.size > 0 else 0
        vol_parietal = ((data[p_x_start:p_x_end, p_y_start:p_y_end, :] >= csf_thresh_22) & (data[p_x_start:p_x_end, p_y_start:p_y_end, :] < tissue_thresh_upper)).sum() * vox_vol
        
        # E. Temporal Atrophy (Axial slice dz = -5, lateral dx_offset = 35, tf = 0.3)
        z_temporal = int(z_com - 5 / spacing[2])
        t_y_start = int(y_com - 30 / spacing[1])
        t_y_end = int(y_com + 30 / spacing[1])
        t_x_l_start = int(x_com - 60 / abs(spacing[0]))
        t_x_l_end = int(x_com - 35 / abs(spacing[0]))
        t_x_r_start = int(x_com + 35 / abs(spacing[0]))
        t_x_r_end = int(x_com + 60 / abs(spacing[0]))
        
        t_roi_l = data[t_x_l_start:t_x_l_end, t_y_start:t_y_end, z_temporal]
        t_roi_r = data[t_x_r_start:t_x_r_end, t_y_start:t_y_end, z_temporal]
        
        csf_sum = 0
        total_size = 0
        if t_roi_l.size > 0:
            csf_sum += (t_roi_l < csf_thresh_30).sum()
            total_size += t_roi_l.size
        if t_roi_r.size > 0:
            csf_sum += (t_roi_r < csf_thresh_30).sum()
            total_size += t_roi_r.size
        ratio_temporal = csf_sum / total_size if total_size > 0 else 0
        
        vol_temporal_l = ((data[t_x_l_start:t_x_l_end, t_y_start:t_y_end, :] >= csf_thresh_30) & (data[t_x_l_start:t_x_l_end, t_y_start:t_y_end, :] < tissue_thresh_upper)).sum() * vox_vol
        vol_temporal_r = ((data[t_x_r_start:t_x_r_end, t_y_start:t_y_end, :] >= csf_thresh_30) & (data[t_x_r_start:t_x_r_end, t_y_start:t_y_end, :] < tissue_thresh_upper)).sum() * vox_vol
        vol_temporal = vol_temporal_l + vol_temporal_r
        
        return {
            'RID': rid,
            'Auto_Lt_Ratio': ratio_hippo_lt,
            'Auto_Rt_Ratio': ratio_hippo_rt,
            'Auto_Front_Ratio': ratio_front,
            'Auto_Parietal_Ratio': ratio_parietal,
            'Auto_Temporal_Ratio': ratio_temporal,
            'Vol_Hippo_Lt': vol_hippo_lt,
            'Vol_Hippo_Rt': vol_hippo_rt,
            'Vol_Front': vol_front,
            'Vol_Parietal': vol_parietal,
            'Vol_Temporal': vol_temporal,
            'Vol_Ventricle': vol_ventricle,
            'Vol_TBV': vol_tbv
        }
    except Exception as e:
        print(f"Error processing RID {rid}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="e-CVRS T1 Atrophy Feature Extraction Pipeline")
    parser.add_argument("--input_dir", type=str, default=".", help="Path to raw MRI folder containing .hdr/.img pairs")
    parser.add_argument("--output_csv", type=str, default="e-CVRS_automated_scores.csv", help="Path to save output features CSV")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
        
    hdr_files = [f for f in os.listdir(args.input_dir) if f.endswith('.hdr')]
    print(f"Found {len(hdr_files)} MRI scans in {args.input_dir}.")
    
    results = []
    
    # Process all scans
    for idx, f in enumerate(hdr_files):
        match = re.search(r'_S_(\d+)_', f)
        if not match:
            continue
        rid = int(match.group(1))
        
        filepath = os.path.join(args.input_dir, f)
        print(f"[{idx+1}/{len(hdr_files)}] Processing RID {rid}...", end='\r')
        
        res = process_subject(filepath, rid)
        if res is not None:
            results.append(res)
            
    print("\nProcessing complete.")
    df_out = pd.DataFrame(results)
    
    # Save the raw ratios and volumes
    df_out.to_csv(args.output_csv, index=False)
    print(f"Automated features successfully saved to {args.output_csv}. Total subjects: {len(df_out)}")

if __name__ == '__main__':
    main()
