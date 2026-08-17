# e-CVRS: Explainable Automated Visual Rating of Brain Atrophy on MRI in Mild Cognitive Impairment 

e-CVRS is a lightweight, pure-Python pipeline that automates the atrophy component of the Comprehensive Visual Rating Scale (CVRS) on 3D T1-weighted MRI — **without FSL, ANTs, FreeSurfer, registration, or segmentation**.

It anchors regions of interest on the anatomical **center of mass (COM)** of the head, extracts local CSF-to-tissue ratios and proxy volumes, and converts them into the familiar CVRS ordinal grades (total 0–17). Every grade ships with a **visual explanation** showing the exact slice, ROI boundaries, and segmented CSF mask that produced it.

Processing takes seconds per scan on a standard CPU.

---

## Published results (v3, adaptive calibration)

Validated in **339 ADNI participants with mild cognitive impairment**, with all model fitting inside training folds and every statistic computed out-of-fold:

| Domain | Reference standard | Result |
| :--- | :--- | :--- |
| Scale fidelity | Blinded expert CVRS ratings | **ICC(2,1) = 0.747** (95% CI 0.694–0.788); regional κw 0.470–0.652; within ±1 grade 90–96% |
| Structural | FreeSurfer v7 hippocampal volume | ρ = 0.384 (manual CVRS: 0.472) |
| Clinical | 6 cognitive outcomes, covariate-adjusted | Significant for **all six** after BH correction (incremental F = 4.2–19.4) |
| Biological | CSF Aβ42 | ρ = −0.286 (manual CVRS: −0.231) |
| Stability | 8 fold-assignment seeds | Total ICC 0.745 ± 0.006 |

The calibration algorithm was selected by a **prespecified equal-weight composite** of the three external domains (structural, clinical, biological), computed strictly on held-out predictions — not by agreement with any single reference standard.

---

## What the model actually does

Each of the five regional grades is predicted from an 11-feature panel — five regional CSF ratios, five proxy volumes normalized by total brain volume, and the global ventricular fraction — in two transparent stages:

1. **Adaptive penalized regression.** A nested four-fold inner cross-validation, run inside each outer training fold, selects *per region* whether feature effects are **linear** (`x`) or **quadratic** (`[x, x²]`), and picks an **L2 ridge penalty** λ ∈ {0, 0.01, 0.1, 1, 10, 100}. The selected model is refit on the full outer training fold.
2. **Ordinal thresholding.** Grade cutoffs maximizing quadratic-weighted kappa are learned from the same training fold and applied unchanged to held-out data.

No latent image representation is learned. The output is an explicit weighted sum of named, inspectable measurements plus explicit thresholds.

**The selection itself is an interpretable result.** Across the 40 outer training folds of the published analysis (8 seeds × 5 folds):

| Region | Linear selected | Quadratic selected |
| :--- | ---: | ---: |
| Left hippocampus MTA | 40 / 40 | 0 / 40 |
| Right hippocampus MTA | 30 / 40 | 10 / 40 |
| Frontal | 1 / 40 | 39 / 40 |
| Parietal | 1 / 40 | 39 / 40 |
| Temporal | 4 / 40 | 36 / 40 |

Hippocampal grading is dominated by a focal, near-linear CSF signal in the choroid fissure; diffuse cortical atrophy accumulates non-linearly across sulci and is better fit by curved effects.

---

## Installation

```bash
git clone https://github.com/neudoc/e-CVRS.git
cd e-CVRS
pip install -r requirements.txt
```

---

## Quick start

Reproduce the published analysis end-to-end with one command:

```bash
./reproduce.sh /path/to/adni/scans
```

Or run the two stages separately:

**1. Extract transparent features from MRI**

```bash
python -m ecvrs.cli extract \
  --input_dir /path/to/scans \
  --output_csv e-CVRS_automated_scores.csv \
  --qc_csv qc_report.csv
```

**2. Calibrate and validate** (the default `adaptive_ridge` is the published method)

```bash
python -m ecvrs.cli evaluate \
  --scores_csv e-CVRS_automated_scores.csv \
  --ratings_excel ADNI_MRI_rating.xlsx \
  --freesurfer_csv UCSF_FS.csv \
  --seed 42 \
  --output_dir results
```

Expected output:

```
Calibration method: adaptive_ridge

Adaptive calibration selections (design, lambda) per outer fold:
  Hippo_Lt     linear 5/5, quadratic 0/5, lambda [10.0, 10.0, 10.0, 10.0, 10.0]
  Hippo_Rt     linear 4/5, quadratic 1/5, lambda [10.0, 10.0, 10.0, 10.0, 10.0]
  Front        linear 0/5, quadratic 5/5, lambda [1.0, 1.0, 10.0, 10.0, 1.0]
  Parietal     linear 0/5, quadratic 5/5, lambda [1.0, 0.1, 1.0, 1.0, 1.0]
  Temporal     linear 0/5, quadratic 5/5, lambda [1.0, 1.0, 1.0, 1.0, 1.0]

Hippo_Lt     Weighted Kappa: 0.6000 [95% CI: 0.5272 - 0.6616]
Hippo_Rt     Weighted Kappa: 0.6219 [95% CI: 0.5592 - 0.6767]
Front        Weighted Kappa: 0.4698 [95% CI: 0.3759 - 0.5539]
Parietal     Weighted Kappa: 0.6312 [95% CI: 0.5584 - 0.6932]
Temporal     Weighted Kappa: 0.6524 [95% CI: 0.5871 - 0.7128]
Total Sum    ICC(2,1):       0.7467 [95% CI: 0.6942 - 0.7884]
```

**3. Render the explanatory overlay for one subject**

```bash
python -m ecvrs.cli render --scan /path/to/subject.hdr --output_dir case_out
```

---

## Calibration methods

`--calibration` selects the mapping from features to ordinal grades:

| Value | Method |
| :--- | :--- |
| `adaptive_ridge` | **Default; the published v3 method.** Nested inner CV selects linear vs. quadratic design and ridge penalty per region. |
| `linear_multi` | v2. Multi-feature ordinary least squares on the same 11-feature panel. |
| `ordinal` | Baseline. Single local CSF ratio, kappa-optimized thresholds. |
| `quantile` | Baseline. Single ratio, cumulative-distribution matching. |

All four are leakage-controlled: thresholds and weights are learned only inside training folds.

---

## Tests

```bash
python tests/test_adaptive_ridge.py
```

Test 1 runs anywhere and verifies that the inner selector recovers a known feature design and applies shrinkage under collinearity. Test 2 reproduces the published agreement statistics exactly, and is skipped automatically when the ADNI-derived inputs are absent.

The `analysis/` scripts reproduce every main-text and supplementary table and figure. Set `ECVRS_DATA_DIR` to the folder holding the ADNI-derived inputs and run, in order: `merge_cohort.py`, `benchmark.py`, `final_analysis.py`, then the supplementary scripts (`grade_fs_analysis.py`, `mta_cognitive_analysis.py` and the two figure scripts).

---

## Repository structure

```
ecvrs/
├── config/
│   └── default.yaml          # ROI offsets, CSF threshold factors, seed
├── io/
│   └── readers.py            # MRI loading and orientation handling
├── preprocess/
│   └── com.py                # Center-of-mass calculation with neck clipping
├── features/
│   ├── csf_ratio.py          # Bounding-box crop and local CSF ratio extraction
│   └── proxy_volume.py       # 3D bounding-box tissue volume proxies
├── calibration/
│   ├── adaptive_ridge.py     # v3 (published): nested linear/quadratic ridge
│   ├── linear_multi.py       # v2: multi-feature OLS
│   ├── ordinal.py            # Kappa-optimized threshold search
│   └── quantile.py           # Cumulative distribution matching
├── stats/
│   ├── agreement.py          # Quadratic-weighted kappa, ICC(2,1), bootstrap CIs
│   ├── regression.py         # OLS, nested F-tests, Benjamini–Hochberg
│   └── validate.py           # Cross-verification against scikit-learn / pingouin
├── viz/
│   └── overlays.py           # Explanatory annotated slice overlays (PNG)
├── qc/
│   └── checks.py             # Dimension checks and out-of-bounds ROI flags
└── cli.py                    # Command-line entry point
analysis/
├── merge_cohort.py           # Build the 339-subject multi-domain cohort
├── benchmark.py              # Candidate-algorithm benchmark (writes out-of-fold grades)
├── benchmark2.py             # Additional candidates
├── final_analysis.py         # Main-text tables (agreement, FS, cognition, CSF)
├── make_figures.py           # Main-text Figures 1-4
├── grade_fs_analysis.py      # Supplementary S5-S6: MTA grades vs FreeSurfer volumes
├── grade_fs_figure.py        # Supplementary Figure S1
├── mta_cognitive_analysis.py # Supplementary S7: bilateral MTA grades vs cognition
└── mta_cognitive_figure.py   # Supplementary Figure S2
tests/
└── test_adaptive_ridge.py    # Selector self-check + published-result reproduction
reproduce.sh                  # One-command end-to-end reproduction
requirements.txt              # Locked environment
```

---

## Using the calibration directly

```python
import numpy as np
from ecvrs.calibration.adaptive_ridge import (
    train_adaptive_ridge_mapping, predict_adaptive_ridge_mapping)

# X_train: (n, 11) feature panel; y_train: manual grades; 5 classes for hippocampal MTA
model = train_adaptive_ridge_mapping(X_train, y_train, num_classes=5, seed=42)

print(model["variant"], model["lam"])      # e.g. 'linear' 10.0
print(model["beta"])                       # explicit standardized coefficients
print(model["thresholds"])                 # explicit grade cutoffs

grades = predict_adaptive_ridge_mapping(X_test, model)
```

`train_adaptive_ridge_mapping` must be called on training data only. `model` is a plain dict of named arrays and can be serialized, inspected, and audited.

---

## Data compliance

The raw T1 MRI scans and clinical spreadsheets analyzed in this study are participant data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**. Under the ADNI Data Use Agreement, **no raw MRI or patient-identifiable clinical data are distributed in this repository**.

To replicate:

1. Apply for access at the [LONI ADNI Portal](https://adni.loni.usc.edu/).
2. Download the screening T1 structural MRI dataset, the UCSF FreeSurfer table, the UW neuropsychological summary, and the UPenn CSF biomarker table.
3. Run the commands above.

---

## Citation

> Jeong HT, Lee S, Youn J, Jang JW, Shin HW, Youn YC, for the Alzheimer's Disease Neuroimaging Initiative. Explainable Automated Visual Rating of Brain Atrophy on MRI in Mild Cognitive Impairment: Development and Multi-Domain Validation of e-CVRS. *(under review)*

Ethics: this secondary analysis of de-identified public data was granted an exemption from full review, with waiver of informed consent, by the Institutional Review Board of Chung-Ang University Hospital (IRB No. 2607-048-19639).

---

## Limitations

- Validated cross-sectionally in a single ADNI MCI cohort scanned at 3T on a harmonized protocol; **external multi-site validation is pending**.
- The COM anchoring tolerates translation but does **not** correct head rotation.
- The neck cutoff is expressed in axial slices (130) rather than physical millimeters; revise for heterogeneous acquisitions.
- e-CVRS tracks CSF Aβ42 but did not reproduce the p-tau181 association shown by volumetry, indicating a sensitivity ceiling of the current fixed-ROI features.
