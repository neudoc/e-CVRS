"""Publication figures for the e-CVRS manuscript (300 dpi)."""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Override with environment variables so the scripts run from a clone
# without editing code:
#   ECVRS_DATA_DIR  directory holding the ADNI-derived inputs
#                   (e-CVRS_automated_scores.csv, ADNI_MRI_rating.xlsx,
#                    UCSF_FS.csv, UWNPSYCHSUM_*.csv, UPENNBIOMK9_*.csv)
#   ECVRS_OUT_DIR   directory for intermediate and final analysis outputs
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("ECVRS_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("ECVRS_OUT_DIR", "analysis_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

o = json.load(open(str(OUT_DIR / 'final_results.json')))
df = pd.read_csv(str(OUT_DIR / 'cohort_merged.csv'))
e42 = pd.read_csv(str(OUT_DIR / 'oof_seed42_adaptive_ridge_nested.csv'))

# palette (colorblind-safe Okabe-Ito subset)
C_ECVRS = '#0072B2'; C_MAN = '#555555'; C_FS = '#D55E00'; C_ACC = '#009E73'
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.color':'#E5E5E5','grid.linewidth':0.6,'axes.axisbelow':True})
FD = str(OUT_DIR / 'figures'); os.makedirs(FD, exist_ok=True)

# ============ Figure 1: pipeline ============
fig, ax = plt.subplots(figsize=(10,5.6)); ax.axis('off'); ax.grid(False)
def box(x,y,w,h,text,fc='#F2F6FA',ec='#0072B2',fs=8.6,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',wrap=True)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color='#333',lw=1.1))
ax.set_xlim(0,1); ax.set_ylim(0,1)
box(0.01,0.72,0.20,0.22,'Baseline 3D T1 MRI\n(ADNI MCI, N=339)\nno registration,\nno segmentation',bold=True)
box(0.26,0.72,0.21,0.22,'Center-of-mass (COM)\nanchoring\nneck stripping;\nphysical-mm ROI offsets')
box(0.52,0.72,0.21,0.22,'Fixed ROI panel\n5 regional CSF ratios\n5 proxy volumes / TBV\nventricular fraction\n(11 transparent features)')
box(0.78,0.72,0.21,0.22,'Explanatory overlay\nslice, ROI box,\nCSF mask, ratio\n(PNG per subject)',ec=C_ACC,fc='#F0FAF6')
arrow(0.21,0.83,0.26,0.83); arrow(0.47,0.83,0.52,0.83); arrow(0.73,0.83,0.78,0.83)
box(0.09,0.38,0.38,0.24,'Adaptive calibration (per region)\nnested 4-fold inner CV chooses:\n• linear or quadratic feature effects\n• ridge penalty λ ∈ {0, 0.01, 0.1, 1, 10, 100}\nrefit on full outer training fold',ec=C_ECVRS,fc='#EAF2FB',bold=False)
box(0.53,0.38,0.30,0.24,'Ordinal thresholding\ngrade cutoffs maximize\nquadratic-weighted κ\n(training fold only)')
arrow(0.62,0.72,0.40,0.62); arrow(0.47,0.50,0.53,0.50)
box(0.09,0.05,0.30,0.22,'e-CVRS grades\nMTA L/R (0–4)\nfrontal/parietal/temporal (0–3)\ntotal atrophy score (0–17)',bold=True)
box(0.46,0.05,0.45,0.22,'Multi-domain validation\nmanual CVRS agreement (κ, ICC)\nFreeSurfer v7 hippocampal volume (ρ)\nsix cognitive outcomes (incremental F, BH)\nCSF Aβ42 / p-tau (ρ)  — all out-of-fold',ec=C_FS,fc='#FDF3EC')
arrow(0.68,0.38,0.30,0.27); arrow(0.39,0.16,0.46,0.16)
ax.text(0.5,0.985,'e-CVRS: COM-anchored feature extraction → adaptive transparent calibration → auditable grades',
        ha='center',va='top',fontsize=10.5,fontweight='bold')
ax.text(0.245,0.66,'5-fold outer CV, all fitting inside training folds (leakage-controlled)',fontsize=8,style='italic',color='#333')
plt.tight_layout(); plt.savefig(f'{FD}/Figure1_pipeline.png',dpi=300,bbox_inches='tight'); plt.close()

# ============ Figure 2: agreement ============
fig, axes = plt.subplots(1,2,figsize=(10,4.2))
ax=axes[0]
regs=[('Hippo_Lt','Left hippocampus MTA'),('Hippo_Rt','Right hippocampus MTA'),('Front','Frontal'),('Parietal','Parietal'),('Temporal','Temporal')]
ys=np.arange(len(regs))[::-1]
for y,(rk,lab) in zip(ys,regs):
    v=o['table2'][rk]
    ax.errorbar(v['kappa'],y,xerr=[[v['kappa']-v['lo']],[v['hi']-v['kappa']]],fmt='o',color=C_ECVRS,capsize=3,ms=6)
    ax.text(v['hi']+0.02,y,f"{v['kappa']:.2f}",va='center',fontsize=8.5)
tv=o['table2']['Total']
ax.errorbar(tv['icc'],-1.2,xerr=[[tv['icc']-tv['lo']],[tv['hi']-tv['icc']]],fmt='s',color=C_FS,capsize=3,ms=7)
ax.text(tv['hi']+0.02,-1.2,f"{tv['icc']:.2f}",va='center',fontsize=8.5,fontweight='bold')
ax.set_yticks(list(ys)+[-1.2]); ax.set_yticklabels([l for _,l in regs]+['Total score ICC(2,1)'])
ax.axvline(0.4,color='#BBB',lw=0.8,ls=':'); ax.axvline(0.6,color='#BBB',lw=0.8,ls=':')
ax.set_xlim(0.3,0.9); ax.set_xlabel('Quadratic-weighted κ / ICC (95% bootstrap CI)')
ax.set_title('A. Agreement with expert manual ratings',loc='left',fontsize=10,fontweight='bold')
ax=axes[1]
m=df['Manual_Atrophy_Sum'].values; e=e42['e_sum'].values
mean=(m+e)/2; dif=e-m
jx=np.random.RandomState(1).uniform(-0.18,0.18,len(m)); jy=np.random.RandomState(2).uniform(-0.18,0.18,len(m))
ax.scatter(mean+jx,dif+jy,s=14,alpha=0.45,color=C_ECVRS,edgecolor='none')
bias=tv['bias']
ax.axhline(bias,color=C_FS,lw=1.4,label=f'Bias {bias:.2f}')
ax.axhline(tv['loa_lo'],color=C_FS,lw=1,ls='--',label=f"95% LoA {tv['loa_lo']:.2f} / {tv['loa_hi']:.2f}")
ax.axhline(tv['loa_hi'],color=C_FS,lw=1,ls='--')
ax.set_xlabel('Mean of e-CVRS and manual total score'); ax.set_ylabel('e-CVRS − manual')
ax.legend(frameon=False,fontsize=8,loc='upper right')
ax.set_title('B. Bland–Altman, total atrophy score (0–17)',loc='left',fontsize=10,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FD}/Figure2_agreement.png',dpi=300,bbox_inches='tight'); plt.close()

# ============ Figure 3: cognitive forest ============
t3=o['table3']
ocs=[('MMSE','MMSE'),('ADAS13','ADAS-Cog-13'),('CDRSB','CDR-SB'),('ADNI_MEM','ADNI-MEM'),('ADNI_EF','ADNI-EF'),('ADNI_LAN','ADNI-LAN')]
flip={'MMSE':-1,'ADAS13':1,'CDRSB':1,'ADNI_MEM':-1,'ADNI_EF':-1,'ADNI_LAN':-1}
# standardize beta by outcome SD to plot on shared axis
sds={k:df[k].std() for k,_ in ocs}
preds=[('manual','Manual CVRS',C_MAN),('ecvrs','e-CVRS',C_ECVRS),('fs','FreeSurfer hippocampus',C_FS)]
fig,ax=plt.subplots(figsize=(7.6,5.6))
yy=0; yticks=[]; ylabs=[]
for k,lab in ocs:
    for j,(pk,plab,col) in enumerate(preds):
        r=t3[k][pk]
        b=r['beta']*flip[k]/sds[k]; se=r['se']/sds[k]
        sig=r['p_bh']<0.05
        ax.errorbar(b,yy,xerr=1.96*se,fmt='o' if pk=='ecvrs' else ('s' if pk=='manual' else 'D'),
                    color=col,mfc=col if sig else 'white',capsize=2.5,ms=6,lw=1.2)
        yy-=1
    yticks.append(yy+2); ylabs.append(lab); yy-=0.8
ax.axvline(0,color='#999',lw=0.9)
ax.set_yticks(yticks); ax.set_yticklabels(ylabs)
ax.set_xlabel('Standardized β (95% CI), oriented so positive = more atrophy → worse cognition')
import matplotlib.lines as mlines
hs=[mlines.Line2D([],[],color=c,marker=mk,ls='None',ms=6,label=l) for (pk,l,c),mk in zip(preds,['s','o','D'])]
ax.legend(handles=hs,frameon=False,fontsize=8.5,loc='lower right')
ax.set_title('Covariate-adjusted association with cognition (filled = BH-adjusted p<0.05)',loc='left',fontsize=10,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FD}/Figure3_cognition.png',dpi=300,bbox_inches='tight'); plt.close()

# ============ Figure 4: CSF ============
t4=o['table4']
fig,ax=plt.subplots(figsize=(7.2,3.6))
groups=[('abeta','CSF Aβ42'),('ptau','CSF p-tau181')]
yy=0; yticks=[]; ylabs=[]
for bmk,glab in groups:
    for pk,plab,col in preds:
        v=t4[f'{pk}_{bmk}']
        rho=v['rho']; lo=v['lo']; hi=v['hi']
        if pk=='fs': rho,lo,hi=-rho,-v['hi'],-v['lo']  # atrophy direction
        sig=v['p']<0.05
        ax.errorbar(rho,yy,xerr=[[rho-lo],[hi-rho]],fmt='o' if pk=='ecvrs' else ('s' if pk=='manual' else 'D'),
                    color=col,mfc=col if sig else 'white',capsize=2.5,ms=6)
        yy-=1
    yticks.append(yy+2); ylabs.append(glab); yy-=0.8
ax.axvline(0,color='#999',lw=0.9)
ax.set_yticks(yticks); ax.set_yticklabels(ylabs)
ax.set_xlabel('Spearman ρ (95% bootstrap CI), atrophy direction (N=321)')
ax.legend(handles=hs,frameon=False,fontsize=8.5,loc='lower right')
ax.set_title('Association with CSF biomarkers: negative Aβ42 / positive p-tau = AD-like (filled = p<0.05)',loc='left',fontsize=9.5,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FD}/Figure4_csf.png',dpi=300,bbox_inches='tight'); plt.close()

# ============ Figure 5: overlay (reuse rendered panel) ============
import shutil
shutil.copy('/mnt/user-data/uploads/CVRS_MCI_APET_ADNI/results/overlays/2010_summary.png', f'{FD}/Figure5_overlay.png')
print('figures done'); print(os.listdir(FD))
