import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def cohen_kappa_quadratic(y_true, y_pred, num_classes):
    O = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        O[int(t), int(p)] += 1
        
    w = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
            
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    E = np.outer(hist_true, hist_pred) / len(y_true)
    
    num = np.sum(w * O)
    den = np.sum(w * E)
    
    if den == 0:
        return 1.0 if num == 0 else 0.0
        
    return 1.0 - (num / den)

def intraclass_correlation(y_true, y_pred):
    targets = np.concatenate([y_true, y_pred])
    n = len(y_true)
    k = 2
    
    grand_mean = np.mean(targets)
    sst = np.sum((targets - grand_mean) ** 2)
    
    subject_means = np.array([np.mean([y_true[i], y_pred[i]]) for i in range(n)])
    ssr = k * np.sum((subject_means - grand_mean) ** 2)
    msr = ssr / (n - 1)
    
    rater_means = np.array([np.mean(y_true), np.mean(y_pred)])
    ssc = n * np.sum((rater_means - grand_mean) ** 2)
    msc = ssc / (k - 1)
    
    sse = sst - ssr - ssc
    mse = sse / ((n - 1) * (k - 1))
    
    icc = (msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse))
    return icc

def calibrate_and_predict_cv(df, ratio_col, manual_col, num_classes):
    # 5-Fold Cross-Validation to learn thresholds and predict held-out ratings
    n = len(df)
    indices = np.arange(n)
    
    # Deterministic shuffle
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    predictions = np.zeros(n, dtype=int)
    ratios = df[ratio_col].values
    manual = df[manual_col].values
    
    for i in range(5):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(5) if j != i])
        
        train_ratios = ratios[train_idx]
        train_manual = manual[train_idx]
        test_ratios = ratios[test_idx]
        
        # Calculate class distribution in training fold
        counts = [np.sum(train_manual == c) for c in range(num_classes)]
        cum_props = np.cumsum(counts) / len(train_manual)
        
        # Compute thresholds on train ratios
        # We skip the last cumulative proportion (which is 1.0)
        thresholds = []
        for p in cum_props[:-1]:
            if p == 0:
                thresholds.append(-1.0) # will assign 0
            else:
                thresholds.append(np.percentile(train_ratios, p * 100))
                
        # Map test ratios using learned thresholds
        test_pred = []
        for r in test_ratios:
            assigned_class = num_classes - 1
            for class_idx, thresh in enumerate(thresholds):
                if r < thresh:
                    assigned_class = class_idx
                    break
            test_pred.append(assigned_class)
            
        predictions[test_idx] = test_pred
        
    return predictions

def main():
    parser = argparse.ArgumentParser(description="e-CVRS Validation and Statistical Analysis Script")
    parser.add_argument("--scores_csv", type=str, default="e-CVRS_automated_scores.csv", help="Path to automated scores CSV")
    parser.add_argument("--ratings_excel", type=str, default="e:\\CVRS_MCI_APET_ADNI\\ADNI_MRI_rating.xlsx", help="Path to manual ratings Excel")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save statistical report and plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.scores_csv):
        print(f"Scores CSV not found: {args.scores_csv}")
        sys.exit(1)
    if not os.path.exists(args.ratings_excel):
        # Try local path fallback
        ratings_local = "ADNI_MRI_rating.xlsx"
        if os.path.exists(ratings_local):
            args.ratings_excel = ratings_local
        else:
            print(f"Ratings Excel not found: {args.ratings_excel}")
            sys.exit(1)
            
    # Load data
    df_auto = pd.read_csv(args.scores_csv)
    df_manual = pd.read_excel(args.ratings_excel)
    
    print(f"Loaded {len(df_auto)} automated subjects and {len(df_manual)} manual subjects.")
    
    # Merge datasets
    df = pd.merge(df_auto, df_manual, on='RID', suffixes=('_auto', '_manual'))
    print(f"Merged dataset contains {len(df)} subjects.")
    
    # Perform 5-Fold Cross-Validation to predict ratings
    df['e_Hippo_Lt'] = calibrate_and_predict_cv(df, 'Auto_Lt_Ratio', 'Hippo_Lt', 5)
    df['e_Hippo_Rt'] = calibrate_and_predict_cv(df, 'Auto_Rt_Ratio', 'Hippo_Rt', 5)
    df['e_Front'] = calibrate_and_predict_cv(df, 'Auto_Front_Ratio', 'Front', 4)
    df['e_Parietal'] = calibrate_and_predict_cv(df, 'Auto_Parietal_Ratio', 'Parietal', 4)
    df['e_Temporal'] = calibrate_and_predict_cv(df, 'Auto_Temporal_Ratio', 'Temporal', 4)
    
    # Calculate totals
    df['Manual_Atrophy_Sum'] = df['Hippo_Lt'] + df['Hippo_Rt'] + df['Front'] + df['Parietal'] + df['Temporal']
    df['e_Atrophy_Sum'] = df['e_Hippo_Lt'] + df['e_Hippo_Rt'] + df['e_Front'] + df['e_Parietal'] + df['e_Temporal']
    
    # Statistical Agreement Analysis
    print("\n=========================================")
    print("ANALYSIS 1: INTER-RATER AGREEMENT (5-Fold Cross-Validated)")
    print("=========================================")
    
    kappa_hippo_l = cohen_kappa_quadratic(df['Hippo_Lt'], df['e_Hippo_Lt'], 5)
    kappa_hippo_r = cohen_kappa_quadratic(df['Hippo_Rt'], df['e_Hippo_Rt'], 5)
    kappa_front = cohen_kappa_quadratic(df['Front'], df['e_Front'], 4)
    kappa_parietal = cohen_kappa_quadratic(df['Parietal'], df['e_Parietal'], 4)
    kappa_temporal = cohen_kappa_quadratic(df['Temporal'], df['e_Temporal'], 4)
    
    icc_total = intraclass_correlation(df['Manual_Atrophy_Sum'].values, df['e_Atrophy_Sum'].values)
    
    print(f"Left Hippocampus Atrophy (0-4) Weighted Kappa: {kappa_hippo_l:.4f}")
    print(f"Right Hippocampus Atrophy (0-4) Weighted Kappa: {kappa_hippo_r:.4f}")
    print(f"Frontal Atrophy (0-3) Weighted Kappa:          {kappa_front:.4f}")
    print(f"Parietal Atrophy (0-3) Weighted Kappa:         {kappa_parietal:.4f}")
    print(f"Temporal Atrophy (0-3) Weighted Kappa:         {kappa_temporal:.4f}")
    print(f"Total Atrophy Sum (0-17) ICC(2,1):             {icc_total:.4f}")
    
    # Volumetry Correlation Analysis
    print("\n=========================================")
    print("ANALYSIS 2: CORRELATION WITH EXPLORATORY PROXY VOLUMES")
    print("=========================================")
    
    # Hippocampus vs. Volume
    r_l_hippo, p_l_hippo = pearsonr(df['Auto_Lt_Ratio'], df['Vol_Hippo_Lt'])
    r_r_hippo, p_r_hippo = pearsonr(df['Auto_Rt_Ratio'], df['Vol_Hippo_Rt'])
    
    # Cortical ratios vs Lobe volumes
    r_front, p_front = pearsonr(df['Auto_Front_Ratio'], df['Vol_Front'])
    r_parietal, p_parietal = pearsonr(df['Auto_Parietal_Ratio'], df['Vol_Parietal'])
    r_temporal, p_temporal = pearsonr(df['Auto_Temporal_Ratio'], df['Vol_Temporal'])
    
    print(f"Left Hippocampus: CSF Ratio vs. 3D Volume:  r = {r_l_hippo:.4f} (p = {p_l_hippo:.2e})")
    print(f"Right Hippocampus: CSF Ratio vs. 3D Volume: r = {r_r_hippo:.4f} (p = {p_r_hippo:.2e})")
    print(f"Frontal Lobe: CSF Ratio vs. 3D Volume:      r = {r_front:.4f} (p = {p_front:.2e})")
    print(f"Parietal Lobe: CSF Ratio vs. 3D Volume:     r = {r_parietal:.4f} (p = {p_parietal:.2e})")
    print(f"Temporal Lobe: CSF Ratio vs. 3D Volume:     r = {r_temporal:.4f} (p = {p_temporal:.2e})")
    
    # Clinical Predictability Analysis
    print("\n=========================================")
    print("ANALYSIS 3: CLINICAL PREDICTABILITY (Correlation with MMSE & CDR-SB)")
    print("=========================================")
    
    df_clean = df.dropna(subset=['MMSE', 'CDRSB', 'ADAS11'])
    print(f"Clean subjects for clinical analysis: {len(df_clean)}")
    
    r_mmse_ecvrs, _ = pearsonr(df_clean['e_Atrophy_Sum'], df_clean['MMSE'])
    r_mmse_manual, _ = pearsonr(df_clean['Manual_Atrophy_Sum'], df_clean['MMSE'])
    
    r_cdrsb_ecvrs, _ = pearsonr(df_clean['e_Atrophy_Sum'], df_clean['CDRSB'])
    r_cdrsb_manual, _ = pearsonr(df_clean['Manual_Atrophy_Sum'], df_clean['CDRSB'])
    
    df_clean = df_clean.copy()
    df_clean['Vol_Hippo_Total'] = df_clean['Vol_Hippo_Lt'] + df_clean['Vol_Hippo_Rt']
    r_mmse_vol, _ = pearsonr(df_clean['Vol_Hippo_Total'], df_clean['MMSE'])
    r_cdrsb_vol, _ = pearsonr(df_clean['Vol_Hippo_Total'], df_clean['CDRSB'])
    
    print(f"Correlation with MMSE (cognitive performance):")
    print(f"  e-CVRS Atrophy Sum:  r = {r_mmse_ecvrs:.4f}")
    print(f"  Manual Atrophy Sum:  r = {r_mmse_manual:.4f}")
    print(f"  3D Hippocampus Vol:  r = {r_mmse_vol:.4f}")
    
    print(f"Correlation with CDR-SB (clinical severity):")
    print(f"  e-CVRS Atrophy Sum:  r = {r_cdrsb_ecvrs:.4f}")
    print(f"  Manual Atrophy Sum:  r = {r_cdrsb_manual:.4f}")
    print(f"  3D Hippocampus Vol:  r = {r_cdrsb_vol:.4f}")
    
    # Save the final results to a summary text file
    summary_path = os.path.join(args.output_dir, "analysis_results.txt")
    with open(summary_path, 'w', encoding='utf-8') as f_out:
        f_out.write("=========================================\n")
        f_out.write("e-CVRS AUTOMATED PIPELINE ANALYSIS REPORT\n")
        f_out.write("=========================================\n\n")
        f_out.write(f"Total Subjects Evaluated: {len(df)}\n\n")
        f_out.write("1. AGREEMENT STATS (5-Fold CV):\n")
        f_out.write(f"  Left Hippo Weighted Kappa: {kappa_hippo_l:.4f}\n")
        f_out.write(f"  Right Hippo Weighted Kappa: {kappa_hippo_r:.4f}\n")
        f_out.write(f"  Frontal Lobe Weighted Kappa: {kappa_front:.4f}\n")
        f_out.write(f"  Parietal Lobe Weighted Kappa: {kappa_parietal:.4f}\n")
        f_out.write(f"  Temporal Lobe Weighted Kappa: {kappa_temporal:.4f}\n")
        f_out.write(f"  Total Atrophy Sum ICC(2,1): {icc_total:.4f}\n\n")
        f_out.write("2. EXPLORATORY PROXY VOLUME CORRELATIONS (r-values):\n")
        f_out.write(f"  Left Hippo CSF Ratio vs 3D Vol:  {r_l_hippo:.4f}\n")
        f_out.write(f"  Right Hippo CSF Ratio vs 3D Vol: {r_r_hippo:.4f}\n")
        f_out.write(f"  Frontal CSF Ratio vs 3D Vol:     {r_front:.4f}\n")
        f_out.write(f"  Parietal CSF Ratio vs 3D Vol:    {r_parietal:.4f}\n")
        f_out.write(f"  Temporal CSF Ratio vs 3D Vol:    {r_temporal:.4f}\n\n")
        f_out.write("3. CLINICAL CORRELATIONS (vs. MMSE / CDRSB):\n")
        f_out.write(f"  e-CVRS vs MMSE:  {r_mmse_ecvrs:.4f} / CDRSB: {r_cdrsb_ecvrs:.4f}\n")
        f_out.write(f"  Manual vs MMSE:  {r_mmse_manual:.4f} / CDRSB: {r_cdrsb_manual:.4f}\n")
        f_out.write(f"  3D Hippo vs MMSE: {r_mmse_vol:.4f} / CDRSB: {r_cdrsb_vol:.4f}\n")
        
    print(f"\nSummary results saved to {summary_path}")
    
    # Generate Plots
    print("\nGenerating statistical plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Scatter plot Hippo CSF Ratio vs 3D Volumetry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(data=df, x='Auto_Lt_Ratio', y='Vol_Hippo_Lt', ax=ax1, color='blue', scatter_kws={'alpha':0.4})
    ax1.set_title('Left Hippocampus: CSF Ratio vs 3D Volumetry')
    ax1.set_xlabel('Automated CSF Ratio')
    ax1.set_ylabel('3D Hippocampus Volume (mm³)')
    
    sns.regplot(data=df, x='Auto_Rt_Ratio', y='Vol_Hippo_Rt', ax=ax2, color='green', scatter_kws={'alpha':0.4})
    ax2.set_title('Right Hippocampus: CSF Ratio vs 3D Volumetry')
    ax2.set_xlabel('Automated CSF Ratio')
    ax2.set_ylabel('3D Hippocampus Volume (mm³)')
    plt.tight_layout()
    plot1_path = os.path.join(args.output_dir, "hippo_ratio_vs_volumetry.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    
    # Plot 2: Atrophy Sum Score: Manual vs Automated
    plt.figure(figsize=(7, 6))
    sns.stripplot(data=df, x='Manual_Atrophy_Sum', y='e_Atrophy_Sum', jitter=0.25, alpha=0.5, color='purple')
    plt.plot([0, 15], [0, 15], color='red', linestyle='--', label='Identity Line')
    plt.title(f'Total Atrophy Sum Score (ICC = {icc_total:.3f})')
    plt.xlabel('Manual Atrophy Sum (0-17)')
    plt.ylabel('Automated e-CVRS Atrophy Sum (0-17)')
    plt.legend()
    plot2_path = os.path.join(args.output_dir, "atrophy_sum_agreement.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    
    # Plot 3: Clinical Predictability (e-CVRS vs MMSE & CDRSB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(data=df_clean, x='e_Atrophy_Sum', y='MMSE', ax=ax1, color='darkred', x_jitter=0.2, scatter_kws={'alpha':0.4})
    ax1.set_title(f'e-CVRS Atrophy Sum vs MMSE (r = {r_mmse_ecvrs:.3f})')
    ax1.set_xlabel('Automated e-CVRS Atrophy Sum (0-17)')
    ax1.set_ylabel('MMSE Score')
    
    sns.regplot(data=df_clean, x='e_Atrophy_Sum', y='CDRSB', ax=ax2, color='darkorange', x_jitter=0.2, scatter_kws={'alpha':0.4})
    ax2.set_title(f'e-CVRS Atrophy Sum vs CDR-SB (r = {r_cdrsb_ecvrs:.3f})')
    ax2.set_xlabel('Automated e-CVRS Atrophy Sum (0-17)')
    ax2.set_ylabel('CDR-SB Score')
    plt.tight_layout()
    plot3_path = os.path.join(args.output_dir, "clinical_correlations.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    
    print(f"Plots generated and saved successfully to {args.output_dir}.")

if __name__ == '__main__':
    main()
