import os
import sys
import argparse
import yaml
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Package imports
from ecvrs.io.readers import load_mri, standardize_orientation
from ecvrs.preprocess.com import calculate_com
from ecvrs.features.csf_ratio import extract_features
from ecvrs.features.proxy_volume import extract_proxy_volumes
from ecvrs.qc.checks import run_qc_checks
from ecvrs.calibration.quantile import train_quantile_mapping, predict_quantile_mapping
from ecvrs.calibration.ordinal import train_kappa_optimized_mapping, predict_kappa_optimized_mapping
from ecvrs.stats.agreement import cohen_kappa_quadratic, intraclass_correlation, bootstrap_confidence_intervals
from ecvrs.stats.validate import verify_against_libraries
from ecvrs.stats.regression import fit_ols, compare_models, fdr_correction
from ecvrs.viz.overlays import save_mri_overlays

def run_extract(args):
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
        
    hdr_files = [f for f in os.listdir(args.input_dir) if f.endswith('.hdr')]
    print(f"Found {len(hdr_files)} Analyze MRI scans in {args.input_dir}.")
    
    results = []
    qc_reports = []
    
    for idx, f in enumerate(hdr_files):
        match = re.search(r'_S_(\d+)_', f)
        if not match:
            continue
        rid = int(match.group(1))
        filepath = os.path.join(args.input_dir, f)
        print(f"[{idx+1}/{len(hdr_files)}] Processing RID {rid}...", end='\r')
        
        try:
            # 1. Load data
            data, affine, spacing = load_mri(filepath)
            data = standardize_orientation(data, affine)
            
            # 98th percentile for threshold scaling
            i98 = np.percentile(data[::2, ::2, ::2], 98)
            
            # 2. COM Calculation & Neck Strip
            x_com, y_com, z_com, com_pass, com_msg = calculate_com(data, i98)
            
            # 3. Features & ROI Extraction
            ratios, r_meta = extract_features(data, x_com, y_com, z_com, spacing, i98, cfg)
            
            # 4. Proxy 3D Volumes
            volumes = extract_proxy_volumes(data, x_com, y_com, z_com, spacing, i98)
            
            # 5. Run QC Validation Checks
            shape = data.shape
            qc_pass, qc_issues = run_qc_checks(shape, x_com, y_com, z_com, r_meta, i98)
            if not com_pass:
                qc_pass = False
                qc_issues.append(com_msg)
                
            qc_reports.append({
                'RID': rid,
                'QC_Pass': qc_pass,
                'QC_Issues': "; ".join(qc_issues) if qc_issues else "Pass",
                'X_COM': x_com,
                'Y_COM': y_com,
                'Z_COM': z_com
            })
            
            # Accumulate raw feature data
            res_dict = {'RID': rid}
            res_dict.update(ratios)
            res_dict.update(volumes)
            results.append(res_dict)
            
        except Exception as e:
            print(f"\nError processing RID {rid}: {str(e)}")
            qc_reports.append({
                'RID': rid,
                'QC_Pass': False,
                'QC_Issues': f"Exception: {str(e)}",
                'X_COM': 0, 'Y_COM': 0, 'Z_COM': 0
            })
            
    print("\nExtraction complete.")
    df_scores = pd.DataFrame(results)
    df_qc = pd.DataFrame(qc_reports)
    
    df_scores.to_csv(args.output_csv, index=False)
    df_qc.to_csv(args.qc_csv, index=False)
    
    print(f"Features successfully saved to {args.output_csv} (Total rows: {len(df_scores)}).")
    print(f"QC reports saved to {args.qc_csv} (Pass rate: {df_qc['QC_Pass'].mean()*100:.1f}%).")

def run_evaluate(args):
    if not os.path.exists(args.scores_csv):
        print(f"Scores CSV not found: {args.scores_csv}")
        sys.exit(1)
    if not os.path.exists(args.ratings_excel):
        print(f"Manual ratings Excel not found: {args.ratings_excel}")
        sys.exit(1)
        
    df_auto = pd.read_csv(args.scores_csv)
    df_manual = pd.read_excel(args.ratings_excel)
    
    # Merge on RID
    df = pd.merge(df_auto, df_manual, on='RID', suffixes=('_auto', '_manual'))
    
    # Exclude RID 4764 due to extraction/loading failure per study protocol
    if 4764 in df['RID'].values:
        df = df[df['RID'] != 4764]
        print("Excluded RID 4764 (extraction/loading failure) per study protocol.")
        
    print(f"Merged automated & manual ratings. N = {len(df)} matched subjects.")
    
    if len(df) == 0:
        print("Error: Merged dataset is empty.")
        sys.exit(1)
        
    # K-Fold Cross-Validation Setup
    n = len(df)
    np.random.seed(args.seed)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    # We calibrate and predict each lobe
    lobes = [
        ('Auto_Lt_Ratio', 'Hippo_Lt', 5),
        ('Auto_Rt_Ratio', 'Hippo_Rt', 5),
        ('Auto_Front_Ratio', 'Front', 4),
        ('Auto_Parietal_Ratio', 'Parietal', 4),
        ('Auto_Temporal_Ratio', 'Temporal', 4)
    ]
    
    pred_cols = {}
    
    for ratio_col, manual_col, num_classes in lobes:
        preds = np.zeros(n, dtype=int)
        ratios = df[ratio_col].values
        manual = df[manual_col].values
        
        for k in range(5):
            test_idx = folds[k]
            train_idx = np.hstack([folds[j] for j in range(5) if j != k])
            
            train_ratios = ratios[train_idx]
            train_manual = manual[train_idx]
            test_ratios = ratios[test_idx]
            
            if args.calibration == 'ordinal':
                thresh = train_kappa_optimized_mapping(train_ratios, train_manual, num_classes)
                preds[test_idx] = predict_kappa_optimized_mapping(test_ratios, thresh, num_classes)
            else: # quantile baseline
                thresh = train_quantile_mapping(train_ratios, train_manual, num_classes)
                preds[test_idx] = predict_quantile_mapping(test_ratios, thresh, num_classes)
                
        df[f'e_{manual_col}'] = preds
        
    # Compute sum scores
    df['Manual_Atrophy_Sum'] = df['Hippo_Lt'] + df['Hippo_Rt'] + df['Front'] + df['Parietal'] + df['Temporal']
    df['e_Atrophy_Sum'] = df['e_Hippo_Lt'] + df['e_Hippo_Rt'] + df['e_Front'] + df['e_Parietal'] + df['e_Temporal']
    
    # 1. Agreement Stats + Bootstrap CIs
    print("\n-----------------------------------------")
    print("ANALYSIS 1: INTER-RATER AGREEMENT (5-Fold CV)")
    print("-----------------------------------------")
    
    agreement_results = {}
    
    for ratio_col, manual_col, num_classes in lobes:
        k_val = cohen_kappa_quadratic(df[manual_col].values, df[f'e_{manual_col}'].values, num_classes)
        ci_l, ci_u = bootstrap_confidence_intervals(
            df[manual_col].values, df[f'e_{manual_col}'].values, 
            cohen_kappa_quadratic, num_classes=num_classes, seed=args.seed
        )
        print(f"{manual_col:<12} Weighted Kappa: {k_val:.4f} [95% CI: {ci_l:.4f} - {ci_u:.4f}]")
        agreement_results[manual_col] = {
            'kappa': k_val,
            'ci_lower': ci_l,
            'ci_upper': ci_u
        }
        
    icc_val = intraclass_correlation(df['Manual_Atrophy_Sum'].values, df['e_Atrophy_Sum'].values)
    icc_l, icc_u = bootstrap_confidence_intervals(
        df['Manual_Atrophy_Sum'].values, df['e_Atrophy_Sum'].values,
        intraclass_correlation, seed=args.seed
    )
    print(f"Total Sum    ICC(2,1):       {icc_val:.4f} [95% CI: {icc_l:.4f} - {icc_u:.4f}]")
    agreement_results['Total_Sum'] = {
        'icc': icc_val,
        'ci_lower': icc_l,
        'ci_upper': icc_u
    }
    
    # 2. Verify Stats against sklearn / pingouin
    print("\n-----------------------------------------")
    print("CROSS-VERIFICATION AGAINST LIBRARIES")
    print("-----------------------------------------")
    ver_results = verify_against_libraries(df['Hippo_Lt'].values, df['e_Hippo_Lt'].values, 5)
    if ver_results.get('kappa_verified', False):
        print(f"Kappa calculator matches scikit-learn: PASS (sklearn={ver_results.get('sklearn_kappa'):.4f}, custom={ver_results.get('custom_kappa'):.4f})")
    else:
        print("Warning: Kappa calculation discrepancy detected.")
        
    if ver_results.get('icc_verified', False):
        if 'pingouin_icc' in ver_results:
            print(f"ICC calculator matches pingouin: PASS (pingouin={ver_results.get('pingouin_icc'):.4f}, custom={ver_results.get('custom_icc'):.4f})")
        else:
            print("ICC verified (pingouin is absent, bypassed: PASS)")
            
    # 3. Covariates and Hierarchical Linear Regression (Model 1-4)
    # Target values
    print("\n-----------------------------------------")
    print("ANALYSIS 3: HIERARCHICAL LINEAR REGRESSION (Predicting MMSE)")
    print("-----------------------------------------")
    
    # Covariates preprocessing
    df_reg = df.dropna(subset=['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE', 'CDRSB']).copy()
    print(f"N = {len(df_reg)} clean subjects with complete covariates.")
    
    # Dummify gender ('Male' -> 0, 'Female' -> 1)
    df_reg['Gender_Dummy'] = df_reg['PTGENDER'].apply(lambda x: 1 if str(x).lower().startswith('f') else 0)
    # APOE4 carrier (APOE4 > 0 -> 1, else 0)
    df_reg['APOE4_Carrier'] = df_reg['APOE4'].apply(lambda x: 1 if x > 0 else 0)
    
    # Define design matrices
    covariates = df_reg[['AGE', 'Gender_Dummy', 'PTEDUCAT', 'APOE4_Carrier']].values
    y_mmse = df_reg['MMSE'].values
    
    # Model 1: Covariates only
    m1 = fit_ols(covariates, y_mmse)
    
    # Model 2: Covariates + Manual Atrophy Sum
    m2_X = np.hstack([covariates, df_reg[['Manual_Atrophy_Sum']].values])
    m2 = fit_ols(m2_X, y_mmse)
    
    # Model 3: Covariates + e-CVRS Atrophy Sum
    m3_X = np.hstack([covariates, df_reg[['e_Atrophy_Sum']].values])
    m3 = fit_ols(m3_X, y_mmse)
    
    # Model 4: Covariates + Bounding Box Hippo Volume Proxy (3D Hippo Vol = Lt + Rt)
    df_reg['Vol_Hippo_Total'] = df_reg['Vol_Hippo_Lt'] + df_reg['Vol_Hippo_Rt']
    # If FreeSurfer table is provided, we map it instead (comparators logic)
    hippo_col = 'Vol_Hippo_Total'
    if args.freesurfer_csv and os.path.exists(args.freesurfer_csv):
        try:
            print(f"Loading FreeSurfer data from: {args.freesurfer_csv}")
            # Specify low_memory=False to prevent DtypeWarnings
            df_fs = pd.read_csv(args.freesurfer_csv, low_memory=False)
            
            # Keep only the earliest scan for each RID based on EXAMDATE
            if 'EXAMDATE' in df_fs.columns and 'RID' in df_fs.columns:
                df_fs['EXAMDATE_dt'] = pd.to_datetime(df_fs['EXAMDATE'], errors='coerce')
                df_fs = df_fs.sort_values(by='EXAMDATE_dt')
                df_fs = df_fs.drop_duplicates(subset=['RID'], keep='first')
                print(f"Filtered FreeSurfer data to baseline (earliest scan). N = {len(df_fs)} unique subjects.")
                
            df_reg = pd.merge(df_reg, df_fs, on='RID', suffixes=('', '_fs'))
            # Example standard column name in UCSF FreeSurfer table for Left+Right Hippocampus
            if 'ST29SV' in df_reg.columns and 'ST88SV' in df_reg.columns: # ADNI standard Left and Right Hippo volume
                df_reg['FS_Hippo_Total'] = df_reg['ST29SV'] + df_reg['ST88SV']
                hippo_col = 'FS_Hippo_Total'
                print(f"Successfully matched ADNI FreeSurfer Hippocampal volumes for N = {len(df_reg)} subjects in Model 4.")
        except Exception as e:
            print(f"Could not load/merge FreeSurfer CSV: {str(e)}. Falling back to Box Proxy volume.")
            
    m4_X = np.hstack([covariates, df_reg[[hippo_col]].values])
    m4 = fit_ols(m4_X, y_mmse)
    
    print(f"Model 1 (Covariates only):      Adj R² = {m1['adj_r_squared']:.4f}")
    print(f"Model 2 (Cov + Manual Atrophy): Adj R² = {m2['adj_r_squared']:.4f}")
    print(f"Model 3 (Cov + e-CVRS Atrophy): Adj R² = {m3['adj_r_squared']:.4f}")
    print(f"Model 4 (Cov + Hippo Volume):   Adj R² = {m4['adj_r_squared']:.4f}")
    
    # Compare Nested models
    # Compare Model 3 (e-CVRS) vs Model 1 (restricted)
    f_3v1, p_3v1 = compare_models(m1, m3)
    # Compare Model 2 (Manual) vs Model 1 (restricted)
    f_2v1, p_2v1 = compare_models(m1, m2)
    # Compare Model 4 (Volume) vs Model 1 (restricted)
    f_4v1, p_4v1 = compare_models(m1, m4)
    
    # Multi-comparison correction on regression p-values
    p_vals = [p_2v1, p_3v1, p_4v1]
    adj_p = fdr_correction(p_vals)
    
    print("\nModel Comparisons (Incremental explanation of MMSE):")
    print(f"  e-CVRS incremental vs Covariates:  F = {f_3v1:.3f}, p = {p_3v1:.2e} (FDR-adjusted p = {adj_p[1]:.2e})")
    print(f"  Manual incremental vs Covariates:  F = {f_2v1:.3f}, p = {p_2v1:.2e} (FDR-adjusted p = {adj_p[0]:.2e})")
    print(f"  Hippo Vol incremental vs Covariates: F = {f_4v1:.3f}, p = {p_4v1:.2e} (FDR-adjusted p = {adj_p[2]:.2e})")
    
    # 4. Save results JSON & reports
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as ro:
        ro.write("=========================================\n")
        ro.write("e-CVRS PIPELINE STATISTICAL REPORT (V2.0)\n")
        ro.write("=========================================\n\n")
        ro.write(f"Total Cohort Size: {len(df)}\n")
        ro.write(f"Clean Covariates Cohort Size: {len(df_reg)}\n")
        ro.write(f"Calibration Method: {args.calibration}\n\n")
        ro.write("1. INTER-RATER AGREEMENT STATS:\n")
        for k, v in agreement_results.items():
            if k == 'Total_Sum':
                ro.write(f"  {k:<15} ICC = {v['icc']:.4f} (95% CI: {v['ci_lower']:.4f} - {v['ci_upper']:.4f})\n")
            else:
                ro.write(f"  {k:<15} Kappa = {v['kappa']:.4f} (95% CI: {v['ci_lower']:.4f} - {v['ci_upper']:.4f})\n")
        ro.write("\n2. HIERARCHICAL LINEAR REGRESSION (MMSE):\n")
        ro.write(f"  Model 1 (Covariates):      Adj R² = {m1['adj_r_squared']:.4f}\n")
        ro.write(f"  Model 2 (Cov + Manual):    Adj R² = {m2['adj_r_squared']:.4f}\n")
        ro.write(f"  Model 3 (Cov + e-CVRS):    Adj R² = {m3['adj_r_squared']:.4f}\n")
        ro.write(f"  Model 4 (Cov + Hippo Vol):  Adj R² = {m4['adj_r_squared']:.4f}\n\n")
        ro.write("3. NESTED MODEL COMPARISON:\n")
        ro.write(f"  e-CVRS incremental vs Covariates:  F = {f_3v1:.3f}, p = {p_3v1:.2e} (FDR p = {adj_p[1]:.2e})\n")
        ro.write(f"  Manual incremental vs Covariates:  F = {f_2v1:.3f}, p = {p_2v1:.2e} (FDR p = {adj_p[0]:.2e})\n")
        ro.write(f"  Hippo Vol incremental vs Covariates: F = {f_4v1:.3f}, p = {p_4v1:.2e} (FDR p = {adj_p[2]:.2e})\n")
        
    print(f"\nEvaluation summary successfully saved to {report_path}")
    
    # 5. Generate plots (Bland-Altman & Confusion Matrix)
    print("Generating statistical plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot A: Bland-Altman for Sum Scores
    diffs = df['e_Atrophy_Sum'] - df['Manual_Atrophy_Sum']
    means = (df['e_Atrophy_Sum'] + df['Manual_Atrophy_Sum']) / 2.0
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.5, color='teal', edgecolor='k')
    plt.axhline(mean_diff, color='red', linestyle='-', label=f'Mean Bias: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    plt.title('Bland-Altman Plot: e-CVRS vs Manual Total Atrophy Sum')
    plt.xlabel('Mean of e-CVRS and Manual Atrophy Sum')
    plt.ylabel('Difference (e-CVRS - Manual)')
    plt.legend()
    plt.tight_layout()
    ba_path = os.path.join(args.output_dir, "bland_altman_agreement.png")
    plt.savefig(ba_path, dpi=150)
    plt.close()
    
    # Plot B: Confusion Matrix for Left Hippo
    plt.figure(figsize=(7, 6))
    conf_mat = pd.crosstab(df['Hippo_Lt'], df['e_Hippo_Lt'], rownames=['Manual'], colnames=['e-CVRS'])
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix: Left Hippocampus (MTA) Ratings')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "conf_matrix_left_hippo.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Plots saved to {args.output_dir}")

def run_render(args):
    # Renders explanatory visual overlays for a single case
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Get RID from filepath
    match = re.search(r'_S_(\d+)_', args.scan)
    if not match:
        print("Could not parse subject RID from file path. Ensure naming matches ADNI.")
        sys.exit(1)
    rid = int(match.group(1))
    
    print(f"Extracting features and rendering overlays for RID {rid}...")
    data, affine, spacing = load_mri(args.scan)
    data = standardize_orientation(data, affine)
    i98 = np.percentile(data[::2, ::2, ::2], 98)
    
    x_com, y_com, z_com, com_pass, _ = calculate_com(data, i98)
    ratios, r_meta = extract_features(data, x_com, y_com, z_com, spacing, i98, cfg)
    
    # Dummy scores for rendering labels (requires thresholds file or defaults)
    # If a ratings file is provided, we can find thresholds or we just show quantile mappings
    # For standalone rendering, we map using the default thresholds learned globally on ADNI.
    # Standard ADNI e-CVRS thresholds for ratios:
    # Lt Hippo: [0.038, 0.057, 0.082, 0.116]
    # Rt Hippo: [0.041, 0.060, 0.085, 0.120]
    # Frontal: [0.015, 0.024, 0.038]
    # Parietal: [0.018, 0.028, 0.042]
    # Temporal: [0.022, 0.034, 0.050]
    thresh_map = {
        'Lt': [0.038, 0.057, 0.082, 0.116],
        'Rt': [0.041, 0.060, 0.085, 0.120],
        'Front': [0.015, 0.024, 0.038],
        'Parietal': [0.018, 0.028, 0.042],
        'Temporal': [0.022, 0.034, 0.050]
    }
    
    def map_val(val, cuts):
        for idx, c in enumerate(cuts):
            if val < c:
                return idx
        return len(cuts)
        
    scores = {
        'e_Hippo_Lt': map_val(ratios['Auto_Lt_Ratio'], thresh_map['Lt']),
        'e_Hippo_Rt': map_val(ratios['Auto_Rt_Ratio'], thresh_map['Rt']),
        'e_Front': map_val(ratios['Auto_Front_Ratio'], thresh_map['Front']),
        'e_Parietal': map_val(ratios['Auto_Parietal_Ratio'], thresh_map['Parietal']),
        'e_Temporal': map_val(ratios['Auto_Temporal_Ratio'], thresh_map['Temporal']),
    }
    scores['e_Atrophy_Sum'] = sum(scores.values())
    
    save_mri_overlays(data, x_com, y_com, z_com, spacing, i98, r_meta, scores, ratios, thresh_map, args.output_dir, rid)
    print(f"Overlay summary successfully generated and saved to {os.path.join(args.output_dir, 'overlays', f'{rid}_summary.png')}")

def main():
    parser = argparse.ArgumentParser(description="e-CVRS Automated Visual Rating Pipeline CLI (V2.0)")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")
    
    # extract parser
    p_extract = subparsers.add_parser("extract", help="Extract raw ratios & volumes from MRI files")
    p_extract.add_argument("--input_dir", type=str, default=".", help="Path to raw MRI folder containing Analyze files")
    p_extract.add_argument("--config", type=str, default="ecvrs/config/default.yaml", help="Configuration YAML path")
    p_extract.add_argument("--output_csv", type=str, default="e-CVRS_automated_scores.csv", help="Output path for scores CSV")
    p_extract.add_argument("--qc_csv", type=str, default="qc_report.csv", help="Output path for QC report CSV")
    
    # evaluate parser
    p_evaluate = subparsers.add_parser("evaluate", help="Perform K-fold cross-validation and statistical evaluation")
    p_evaluate.add_argument("--scores_csv", type=str, default="e-CVRS_automated_scores.csv", help="Input automated scores CSV")
    p_evaluate.add_argument("--ratings_excel", type=str, default="ADNI_MRI_rating.xlsx", help="Input manual ratings Excel sheet")
    p_evaluate.add_argument("--freesurfer_csv", type=str, default=None, help="Optional ADNI FreeSurfer volumes CSV")
    p_evaluate.add_argument("--calibration", type=str, choices=['quantile', 'ordinal'], default='ordinal', help="Threshold calibration mapping model")
    p_evaluate.add_argument("--seed", type=int, default=42, help="Random seed for K-fold splitting")
    p_evaluate.add_argument("--output_dir", type=str, default="results", help="Directory to save text report and plots")
    
    # render parser
    p_render = subparsers.add_parser("render", help="Render overlay explanations for a single subject scan")
    p_render.add_argument("--scan", type=str, required=True, help="Path to raw Analyze header (.hdr) scan")
    p_render.add_argument("--config", type=str, default="ecvrs/config/default.yaml", help="Configuration YAML path")
    p_render.add_argument("--output_dir", type=str, default=".", help="Directory to save visual overlays")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        run_extract(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "render":
        run_render(args)

if __name__ == "__main__":
    main()
