# e-CVRS: Automated & Explainable Visual Rating Scale for Brain Atrophy

e-CVRS is a lightweight, pure-Python pipeline designed to automate visual rating scales for brain atrophy on 3D T1-weighted MRI scans without external neuroimaging suite dependencies (such as FSL, ANTs, or FreeSurfer). 

By automatically calculating the anatomical Center of Mass (COM) of the brain and defining regions of interest (ROIs) relative to this coordinate, e-CVRS extracts local CSF-to-tissue ratios to assign visual atrophy scores. Additionally, it generates **explanatory visual overlays** to illustrate the anatomical rationale behind each score.

---

## Key Features

1. **Explainable AI (Interpretable-by-design):** Generates PNG overlays highlighting the exact slice, ROI box boundaries, and segmented CSF mask used to determine the atrophy score.
2. **Zero Heavy Dependencies:** Written entirely in Python using standard scientific libraries (`nibabel`, `numpy`, `scipy`, `pandas`, `matplotlib`). No registration or segmentation suites required.
3. **Data Leakage Protected:** Maps continuous CSF ratios to discrete visual rating classes (MTA 0-4, Cortical 0-3) using cross-validated threshold calibration (optimizing quadratic weighted Kappa strictly inside training splits).
4. **FDR-Corrected OLS Regressions:** Includes statistical validation scripts for inter-rater agreement (weighted Kappa, ICC with 95% Bootstrap CIs) and clinical MMSE predictability (hierarchical linear regression).

---

## Repository Structure

```
ecvrs/
├── config/
│   └── default.yaml          # ROI offsets, CSF threshold scaling factors, seed
├── io/
│   ├── readers.py            # MRI loading and orientation check (LPS/RAS)
│   └── writers.py            # CSV and report writer utilities
├── preprocess/
│   ├── com.py                # Centroid COM calculation with neck clipping
│   └── reorient.py           # Standard coordinate reorientation helpers
├── features/
│   ├── csf_ratio.py          # Bounding box crop and local CSF ratio extraction
│   └── proxy_volume.py       # 3D bounding box tissue volume proxy (internal metrics)
├── calibration/
│   ├── quantile.py           # Cumulative distribution matching calibration
│   └── ordinal.py            # Simplex Kappa-optimization calibration
├── stats/
│   ├── agreement.py          # Cohen's quadratic weighted Kappa, ICC(2,1)
│   ├── regression.py         # OLS regressions and hierarchical F-tests
│   └── validate.py           # Stats cross-verification (vs. scikit-learn)
├── viz/
│   └── overlays.py           # Explanatory annotated slice overlays (PNG)
├── qc/
│   └── checks.py             # Dimension checks and out-of-bounds ROI flags
└── cli.py                    # Main CLI parser entry point
```

---

## Installation & Setup

1. Clone this repository (local code setup):
   ```bash
   git clone https://github.com/<username>/e-CVRS.git
   cd e-CVRS
   ```

2. Install python scientific libraries:
   ```bash
   pip install nibabel numpy pandas scipy matplotlib seaborn pyyaml openpyxl
   ```

---

## Command Line Interface (CLI)

The package exposes a command-line interface via `ecvrs.cli`.

### 1. Extract Local Ratios
Processes a directory of raw Analyze (`.hdr`/`.img`) or NIfTI scans to calculate the COM, extract localized CSF ratios, and output a feature sheet and a QC audit report:
```bash
python -m ecvrs.cli extract --input_dir /path/to/scans --output_csv e-CVRS_automated_scores.csv --qc_csv qc_report.csv
```

### 2. Statistical Validation
Merges the extracted automated features with a manual clinical spreadsheet, performs 5-fold cross-validation calibration, computes inter-rater agreement (Kappa, ICC with 95% CIs), verifies math against libraries, and fits hierarchical nested linear regressions to predict MMSE:
```bash
python -m ecvrs.cli evaluate --scores_csv e-CVRS_automated_scores.csv --ratings_excel ADNI_MRI_rating.xlsx --output_dir results
```

### 3. Render Visual Overlay for a Single Subject
Extracts features and exports an explanatory bounding box / CSF mask summary PNG panel for a specific scan:
```bash
python -m ecvrs.cli render --scan /path/to/subject.hdr --output_dir case_out
```

---

## Data Compliance Note
The raw T1 MRI scans and clinical spreadsheets analyzed in this study are proprietary participant data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**. In compliance with the ADNI Data Use Agreement, **no raw MRIs or patient-identifiable clinical data are distributed in this repository**. 

To replicate the experiment:
1. Apply for database access at the [LONI ADNI Portal](https://adni.loni.usc.edu/).
2. Download the screening T1 structural MRI dataset.
3. Place files in the working directory and run the commands above.
