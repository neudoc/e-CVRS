# e-CVRS: Explainable Automated Visual Rating of Brain Atrophy on MRI in Mild Cognitive Impairment

e-CVRS is a lightweight, pure-Python pipeline that automates the atrophy component of the Comprehensive Visual Rating Scale (CVRS) on 3D T1-weighted MRI — **without FSL, ANTs, FreeSurfer, registration, or segmentation**.

It anchors regions of interest on the anatomical **center of mass (COM)** of the head, extracts local CSF-to-tissue ratios and proxy volumes, and converts them into the familiar CVRS ordinal grades (total 0–17). Every grade ships with a **visual explanation** showing the exact slice, window boundaries, and segmented CSF mask that produced it.

Processing takes seconds per scan on a standard CPU.

---

## Published results

Validated in **339 ADNI participants with mild cognitive impairment**, with all model fitting inside training folds and every statistic computed out-of-fold:

| Domain | Reference standard | Result |
| :--- | :--- | :--- |
| Scale fidelity | Blinded expert CVRS ratings | **ICC(2,1) = 0.748** (95% CI 0.696–0.793); regional κw 0.498–0.666; within ±1 grade 90–99% |
| Structural | FreeSurfer v7 hippocampal volume | ρ = 0.416 (manual CVRS: 0.472) |
| Clinical | 6 cognitive outcomes, covariate-adjusted | Significant for **all six** after BH correction (incremental F = 5.7–24.7), with the **largest executive-function increment of the three imaging measures compared** |
| Biological | CSF Aβ42 | ρ = −0.282, **stronger than the expert reading** (−0.231) |
| Stability | 8 fold-assignment seeds | Total ICC 0.734 ± 0.011 (range 0.719–0.748) |

The calibration algorithm was selected by a **prespecified equal-weight composite** of the three external domains (structural, clinical, biological), computed strictly on held-out predictions — not by agreement with any single reference standard.

---

## Where the medial temporal grades come from

The defining design choice is that the medial temporal grades are computed from **windows anchored on the hippocampal region**, i.e. on the anatomy the MTA scale actually describes.

For each hemisphere the CSF ratio is measured in a **medial band 14–34 mm lateral to the midline and 25 mm tall, centered 18 mm below the COM plane** — directly over the temporal horn, choroid fissure, and parahippocampal CSF spaces whose widening constitutes the Scheltens grade. The ratio is **averaged over 31 contiguous coronal slices**, spanning `y_com − 5 mm` (hippocampal tail) to `y_com + 25 mm` (head), so no single-slice placement error can dominate the measurement. Window placement was verified against hippocampal anatomy by a neurologist across the atrophy range.

The same along-axis averaging is applied to every window in the pipeline: each **lobar CSF ratio is averaged over 11 contiguous axial slices** rather than read from a single plane.

That the window measures the intended anatomy is testable, and it tests out: the **raw, uncalibrated** medial temporal CSF ratio correlates with FreeSurfer hippocampal volume at ρ = 0.413 (left) and 0.358 (right) — the left band alone matching the fully calibrated total score.

A **tilted** variant whose vertical center follows the hippocampal long axis (12 mm below the COM plane at the tail to 20 mm at the head) was evaluated under the identical protocol. It improved the raw feature–grade correlation (ρ = 0.46 vs 0.43) but won no external validity domain, and the simpler fixed-height window was retained by the prespecified rule.

---

## What the model actually does

Each of the five regional grades is predicted from an 11-feature panel — five regional CSF ratios, five proxy volumes normalized by total brain volume, and the global ventricular fraction — in two transparent stages:

1. **Quadratic-basis ridge regression.** The panel is expanded into an explicit quadratic basis `[x, x²]` and fitted by ridge regression, with the penalty λ ∈ {0, 0.01, 0.1, 1, 10, 100} selected by four-fold inner cross-validation run inside each outer training fold.
2. **Ordinal thresholding.** Grade cutoffs maximizing quadratic-weighted kappa are learned from the same training fold and applied unchanged to held-out data.

No latent image representation is learned. The output is an explicit weighted sum of named, inspectable measurements plus explicit thresholds.

### How the calibration was chosen

Five transparent candidates were compared across 8 fold-assignment seeds and ranked by the equal-weight composite of the three external domains (lower rank = better):

| Candidate | ICC | FreeSurfer ρ | Mean cognitive F | Mean \|ρ\| CSF | **Composite rank** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`quadratic_ridge`** (published) | 0.734 | 0.405 | 15.65 | 0.167 | **1.67** |
| `adaptive_ridge` | 0.729 | 0.403 | 15.61 | 0.164 | 2.29 |
| linear ridge | 0.704 | 0.369 | 15.15 | 0.160 | 3.08 |
| linear OLS | 0.702 | 0.366 | 14.77 | 0.162 | 3.38 |
| rank-transformed OLS | 0.705 | 0.352 | 13.72 | 0.150 | 4.58 |

**The selected penalties are themselves an interpretable result.** Across the 40 outer training folds (8 seeds × 5 folds), moderate shrinkage predominated, with markedly stronger shrinkage for the medial temporal grades:

| Region | λ=0.01 | λ=0.1 | λ=1 | λ=10 | λ=100 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Left hippocampus MTA | 0 | 1 | 19 | 13 | 7 |
| Right hippocampus MTA | 1 | 5 | 10 | 9 | 15 |
| Frontal | 0 | 2 | 30 | 7 | 1 |
| Parietal | 0 | 15 | 24 | 1 | 0 |
| Temporal | 1 | 8 | 29 | 1 | 1 |

The cortical grades settle on λ=1 as the modal choice; the medial temporal grades, whose focal CSF signal is measured against a highly collinear whole-brain panel, are pushed toward much heavier regularization.

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

Or run the stages separately:

**1. Extract transparent features from MRI**

```bash
python -m ecvrs.cli extract \
  --input_dir /path/to/scans \
  --output_csv e-CVRS_automated_scores.csv \
  --qc_csv qc_report.csv
```

**2. Calibrate and validate** (the default `quadratic_ridge` is the published method)

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
Calibration method: quadratic_ridge

Hippo_Lt     Weighted Kappa: 0.5987 [95% CI: 0.5285 - 0.6620]
Hippo_Rt     Weighted Kappa: 0.6225 [95% CI: 0.5541 - 0.6801]
Front        Weighted Kappa: 0.4981 [95% CI: 0.4099 - 0.5757]
Parietal     Weighted Kappa: 0.5944 [95% CI: 0.5152 - 0.6620]
Temporal     Weighted Kappa: 0.6655 [95% CI: 0.6061 - 0.7212]
Total Sum    ICC(2,1):       0.7482 [95% CI: 0.6961 - 0.7930]
```

**3. Render the explanatory overlay for one subject**

```bash
python -m ecvrs.cli render --scan /path/to/subject.hdr --output_dir case_out
```

---

## Window geometry

All offsets are **physical millimetres** from the head center of mass, so they transfer across voxel sizes without registration. Orientation convention is LAS (`+x` = right, `+y` = anterior, `+z` = superior), as delivered by ADNI Analyze 7.5.

| Window | Placement | Threshold | Sampling |
| :--- | :--- | :--- | :--- |
| Medial temporal (L/R) | 14–34 mm lateral; 25 mm tall; center `z_com − 18 mm` | 0.25·I₉₈ | mean over 31 coronal slices, `y_com − 5` → `+25 mm` |
| MT volume proxy | ±24 mm lateral (35 mm wide); 20 mm deep at `y_com + 10 mm`; same depth/height | 0.25–0.95·I₉₈ | 3D box |
| Frontal | `z_com + 20 mm`, anterior | 0.30·I₉₈ | mean over 11 axial slices |
| Parietal | `z_com + 35 mm`, posterior | 0.20·I₉₈ | mean over 11 axial slices |
| Temporal | `z_com − 5 mm`, bilateral lateral bands | 0.30·I₉₈ | mean over 11 axial slices |

Every value lives in [`ecvrs/config/default.yaml`](ecvrs/config/default.yaml) and can be changed without touching code.

---

## Calibration methods

`--calibration` selects the mapping from features to ordinal grades:

| Value | Method |
| :--- | :--- |
| `quadratic_ridge` | **Default; the published method.** Explicit quadratic basis `[x, x²]` with a ridge penalty chosen by inner CV. |
| `adaptive_ridge` | Additionally selects linear vs. quadratic design per region by nested inner CV. Runner-up in the benchmark. |
| `linear_multi` | Multi-feature ordinary least squares on the same 11-feature panel. |
| `ordinal` | Baseline. Single local CSF ratio, kappa-optimized thresholds. |
| `quantile` | Baseline. Single ratio, cumulative-distribution matching. |

All are leakage-controlled: thresholds and weights are learned only inside training folds.

---

## Tests

```bash
python tests/test_calibration.py
```

Test 1 runs anywhere and verifies that the penalty selector shrinks under collinearity, that the quadratic design is the literal basis `[x, x²]`, and that a trained model exposes its coefficients and cutoffs as plain arrays. Test 2 reproduces the published agreement statistics exactly, and is skipped automatically when the ADNI-derived inputs are absent.

---

## Repository structure

```
ecvrs/
├── config/
│   └── default.yaml          # window geometry, CSF threshold factors, seed
├── io/
│   └── readers.py            # MRI loading and orientation handling
├── preprocess/
│   └── com.py                # center-of-mass calculation with neck clipping
├── features/
│   ├── csf_ratio.py          # hippocampus-anchored + lobar multi-slice CSF ratios
│   └── proxy_volume.py       # 3D bounding-box tissue volume proxies
├── calibration/
│   ├── quadratic_ridge.py    # published: explicit [x, x²] basis + ridge
│   ├── adaptive_ridge.py     # runner-up: nested linear/quadratic selection
│   ├── linear_multi.py       # multi-feature OLS
│   ├── ordinal.py            # kappa-optimized threshold search
│   └── quantile.py           # cumulative distribution matching
├── stats/
│   ├── agreement.py          # quadratic-weighted kappa, ICC(2,1), bootstrap CIs
│   ├── regression.py         # OLS, nested F-tests, Benjamini–Hochberg
│   └── validate.py           # cross-verification against scikit-learn / pingouin
├── viz/
│   └── overlays.py           # explanatory annotated slice overlays (PNG)
├── qc/
│   └── checks.py             # dimension checks and out-of-bounds window flags
└── cli.py                    # command-line entry point
analysis/
├── merge_cohort.py           # build the 339-subject multi-domain cohort
├── benchmark.py              # 5-candidate benchmark + prespecified composite ranking
├── final_analysis.py         # main-text Tables 1–5
└── make_figures.py           # main-text Figures 1–4
tests/
└── test_calibration.py       # selector self-check + published-result reproduction
reproduce.sh                  # one-command end-to-end reproduction
requirements.txt              # locked environment
```

The `analysis/` scripts read their inputs from `ECVRS_DATA_DIR` and write to `ECVRS_OUT_DIR` (figures to `ECVRS_FIG_DIR`), so they run from a clone without editing code. Run them in order: `merge_cohort.py`, `benchmark.py`, `final_analysis.py`, `make_figures.py`.

---

## Using the calibration directly

```python
import numpy as np
from ecvrs.calibration.quadratic_ridge import (
    train_quadratic_ridge_mapping, predict_quadratic_ridge_mapping)

# X_train: (n, 11) feature panel; y_train: manual grades; 5 classes for hippocampal MTA
model = train_quadratic_ridge_mapping(X_train, y_train, num_classes=5, seed=42)

print(model["lam"])                        # ridge penalty chosen by inner CV
print(model["beta"])                       # 22 explicit standardized coefficients [x, x²]
print(model["thresholds"])                 # explicit grade cutoffs

grades = predict_quadratic_ridge_mapping(X_test, model)
```

`train_quadratic_ridge_mapping` must be called on training data only. `model` is a plain dict of named arrays and can be serialized, inspected, and audited.

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

## Version history

- **Current (hippocampus-anchored).** Medial temporal windows placed on the hippocampal region and averaged over 31 coronal slices; lobar windows averaged over 11 axial slices; `quadratic_ridge` calibration. This is the version described in the manuscript under review.
- **`v3-peri-sylvian` (tag).** An earlier development version whose medial temporal window sat at `z_com − 12 mm`, at peri-Sylvian/insular rather than hippocampal level, calibrated with `adaptive_ridge`. Preserved as a git tag for the development record only; **it is not the submitted analysis** and its reported numbers differ.

---

## Limitations

- Validated cross-sectionally in a single ADNI MCI cohort scanned at 3T on a harmonized protocol; **external multi-site validation is pending**.
- The COM anchoring tolerates translation but does **not** correct head rotation.
- Because the hippocampal tail ascends posteriorly, at the most posterior levels of the averaging range the fixed-height window samples adjacent parahippocampal and peritrigonal medial temporal CSF — consistent with the medial temporal, rather than hippocampus-exclusive, construct of the MTA grade.
- The neck cutoff is expressed in axial slices (130) rather than physical millimetres; revise for heterogeneous acquisitions.
- The lobar slice count is fixed at 11 slices, which corresponds to ±5 mm only at the 1.0–1.05 mm slice spacing of the analyzed data; rescale for markedly thicker slices.
- e-CVRS tracks CSF Aβ42 but did not reproduce the p-tau181 association shown by volumetry, indicating a sensitivity ceiling of the current fixed-window features.
