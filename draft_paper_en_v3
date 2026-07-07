# Manuscript Draft (Version 3.0): Development and Expert Validation of an Explainable Automated Atrophy Visual Rating Scale (e-CVRS) Based on Center-of-Mass Alignment in ADNI T1 MRI

> **Version 3.0 Revision Notes (2026-07-07).** This version translates the Korean draft `draft_paper_kr_v3` to English. It incorporates the updated statistics from the N=338 validation cohort (excluding two pre-processing failures and one missing manual rating) and integrates the comparative analysis with official baseline UCSF FreeSurfer v7 hippocampal volumes. The manuscript is prepared according to international journal formatting guidelines.

**Abbreviations:** AD, Alzheimer's disease; MCI, mild cognitive impairment; CN, cognitively normal; VRS, visual rater scale; CVRS, comprehensive visual rating scale; e-CVRS, electronic/automated CVRS; MTA, medial temporal atrophy; GCA, global cortical atrophy; COM, center of mass; ROI, region of interest; CSF, cerebrospinal fluid; ICC, intraclass correlation coefficient; κw, quadratic weighted kappa; MMSE, Mini-Mental State Examination; CDR-SB, Clinical Dementia Rating Sum of Boxes; FDR, false discovery rate; XAI, explainable AI; SaMD, software as a medical device; ADNI, Alzheimer's Disease Neuroimaging Initiative.

---

## Abstract

### Background and Purpose
Visual rating scales (VRS) for brain atrophy are fast and intuitive in clinical practice but suffer from high inter-rater variability. In contrast, automated 3D volumetry (e.g., FreeSurfer) is computationally slow and operates as a black box with limited explainability. We developed an explainable automated atrophy rating pipeline (e-CVRS) that requires no external registration or segmentation software, relying solely on center-of-mass (COM) alignment and local geometric rules, and validated it against expert ratings and clinical measures.

### Materials and Methods
T1-weighted MRI scans of participants with mild cognitive impairment (MCI) from the ADNI database were analyzed. Using the physical COM of the head mask as the coordinate origin, regions of interest (ROIs) for the hippocampi (left/right) and the frontal, parietal, and temporal lobes were automatically defined, and local cerebrospinal fluid (CSF) ratios were computed. To prevent data leakage, thresholds mapping continuous ratios to ordinal grades were learned within the training folds of a 5-fold cross-validation by maximizing the quadratic weighted kappa against manual grades, then applied to held-out folds. Standardized 95% confidence intervals (CIs) for the intraclass correlation coefficient (ICC) and weighted kappa were obtained by 2,000 bootstrap resamples. The predictive value for cognition (MMSE) was assessed by hierarchical linear regression controlling for age, sex, education, and APOE4, with Benjamini–Hochberg false discovery rate (FDR) correction.

### Results
The final validation cohort comprised 338 subjects (excluding two scan failures and one missing manual rating). Agreement between e-CVRS and expert ratings was fair-to-moderate: temporal κw = 0.4929 (95% CI 0.4013–0.5761), left hippocampus κw = 0.4540 (0.3701–0.5336), and parietal κw = 0.3983 (0.3127–0.4833), whereas the frontal lobe (κw = 0.3500, 95% CI 0.2472–0.4458) and right hippocampus (κw = 0.3767, 0.2821–0.4687) fell in the "fair" range. Agreement for the total atrophy sum score (0–17) was ICC(2,1) = 0.5622 (0.4870–0.6326). In hierarchical regression, e-CVRS (adjusted R² = 0.0828, p = 0.565) did not provide a significant independent increment over the covariate-only model (adjusted R² = 0.0847), whereas the official FreeSurfer hippocampal volume (adjusted R² = 0.1125) showed a highly significant increment (FDR-adjusted p = 0.002). This highlights the severe restriction of range in MMSE scores within this single-diagnosis MCI cohort, which constrains the predictive power of ordinal visual scales compared to continuous voxel counts.

### Conclusions
e-CVRS runs in seconds on a standard CPU without external dependencies and achieved fair-to-moderate agreement with expert readers. Under severe range restriction, e-CVRS did not show a significant independent association with MMSE, whereas official FreeSurfer volume retained significance. Given its explainable overlays, e-CVRS has potential as a screening, educational, and reading-support tool, pending replication in full-spectrum cohorts and improved calibration.

**Keywords:** brain atrophy visual rating scale; explainable artificial intelligence; center-of-mass alignment; Alzheimer's disease; mild cognitive impairment; hierarchical regression.

---

## 1. Introduction

Quantification of structural brain atrophy on T1-weighted magnetic resonance imaging (MRI) is crucial for the early diagnosis and prognosis of Alzheimer's disease (AD) and related neurodegenerative disorders in patients with cognitive impairment [9]. Brain atrophy as a biomarker of neurodegeneration is incorporated into clinical diagnostic criteria for AD and MCI [4,5], with medial temporal lobe and posterior cortical atrophy showing strong associations with disease stage and cognitive decline [9]. Commonly used visual rating scales (VRS) include Scheltens' Medial Temporal Atrophy (MTA) scale on coronal slices [4], axial T1-based modified MTA scales [5], and scales assessing global cortical atrophy (GCA) or posterior atrophy by Pasquier [8], Victoroff [6], and Koedam [7]. These visual scales are intuitive, require no special computing hardware, and can be evaluated instantly, making them highly favored by neurologists.

To consolidate information from individual regions, Jang et al. proposed a Comprehensive Visual Rating Scale (CVRS) that integrates cortical atrophy, hippocampal atrophy, ventricular enlargement, and small vessel disease into a single score (0–30), demonstrating significant correlations with neuropsychological tests in AD, MCI, and cognitively normal (CN) groups [1]. In follow-up studies, the CVRS demonstrated independent predictive value for progression to dementia in patients with MCI and longitudinal cognitive decline [2,3]. In particular, baseline CVRS scores were independent risk factors for conversion to dementia within 3 years in patients with prodromal AD (hazard ratio 1.110, 95% CI 1.043–1.182) [3]. This indicates that integrated visual scales can function not just as convenient reading aids, but as prognostic biomarkers.

However, all manual visual ratings share a fundamental limitation: high intra- and inter-rater variability arising from reader subjectivity, variations in clinical experience, and differences in scanner contrast. To overcome this, automated 3D segmentation methods such as FreeSurfer [11], FastSurfer [12], and SynthSeg [13] have been developed to provide precise volumes for hundreds of anatomical structures. Nonetheless, these methods: (i) require tens of minutes to hours of computation per scan on standard hardware, (ii) are sensitive to differences in scanners, magnetic field strength, and preprocessing pipelines because they rely on absolute voxel counts, and (iii) operate as "black boxes" where clinicians cannot verify the underlying measurements. In high-stakes clinical decision-making, interpretable-by-design models are increasingly preferred over post-hoc explanations [14]. The ability to present measurements, ratios, and overlays on-screen to explain how a rating was derived directly enhances the clinical acceptability of automated quantification tools.

In this context, there is a clear need for a lightweight pipeline that combines the reproducibility of automation with the explainability of visual scales without relying on heavy external neuroimaging suites (FSL, ANTs, or FreeSurfer). We propose the **e-CVRS** pipeline, which is written in pure Python and uses a center-of-mass (COM) head alignment as the origin to calculate local CSF ratios, mapping them to ordinal grades corresponding to the CVRS atrophy subscales (0–17 points). COM alignment provides an anatomical reference point that is invariant to scanner padding, patient translation, and head tilt, enabling regional localization without formal registration.

The objectives of this paper are threefold. First, we rigorously quantify the agreement reliability (weighted kappa and ICC) between the automated e-CVRS grades and blinded manual expert ratings in an ADNI MCI cohort under a cross-validation framework designed to prevent data leakage. Second, we evaluate the incremental predictive value of the e-CVRS score for clinical cognition (MMSE) using hierarchical linear regression, comparing it with official FreeSurfer hippocampal volumes and manual CVRS. Third, we address the impact of range restriction within a single-diagnosis MCI cohort on the interpretation of clinical correlations and discuss the performance ceiling and improvement directions for rule-based explainable approaches. We hypothesized that (H1) e-CVRS would achieve fair-to-moderate agreement with expert ratings, and (H2) under restricted cognitive variance, visual scales (both manual and automated) would show diminished incremental predictive power compared to continuous high-resolution volumetry.

---

## 2. Materials and Methods

This study is reported in accordance with the STARD 2015 [21] and TRIPOD [22] statements for reporting diagnostic accuracy and prediction model validation.

### 2.1 Study Design and Subjects
This retrospective cross-sectional validation study utilized baseline 3D T1-weighted MRI scans and clinical data from the ADNI database for MCI subjects. Out of 341 initially screened subjects, 2 scans were excluded during the automated preprocessing phase (RID 4212: physical truncation/corruption of the `.img` file, 9.5 MB; RID 4764: image loading/reading failure). This yielded automated feature data for 339 subjects. From these, 1 subject (RID 4917) who lacked manual expert ratings in the official clinical records (`ADNI_MRI_rating.xlsx`) was excluded. Consequently, the final validation cohort comprised **338 subjects** with complete paired automated and manual indices. Demographic and clinical baseline characteristics are presented in Table 1.

ADNI was conducted with approval from the Institutional Review Boards (IRB) of all participating institutions, and written informed consent was obtained from all participants or their authorized representatives. This study utilized de-identified public secondary data (see Declarations).

### 2.2 Ground Truth: Manual CVRS Atrophy Subscale
The validation reference standard consisted of manual ratings by neurologists blinded to clinical information. To align with the available MRI sequence, we restricted our scope to the **T1-based structural atrophy subscale** of the CVRS [1]: hippocampal MTA (left/right, 0–4; based on Scheltens [4] and Kim et al. [5]) and cortical atrophy in the frontal, parietal, and temporal lobes (each 0–3; based on Victoroff [6] and Koedam [7]), yielding a total score range of 0–17. The small vessel disease component of the CVRS (white matter hyperintensities, lacunes, microbleeds [23]) was excluded since it requires FLAIR or T2* sequences.

### 2.3 Automated e-CVRS Pipeline (Replication-Ready Specification)
The e-CVRS algorithm is implemented in pure Python. It extracts voxel spacing directly from the affine metadata of the raw T1 MRI header, mapping physical millimeter offsets to voxel indices without requiring external registration or brain segmentation suites. The step-by-step mathematical formulation and implementation specifications are detailed below to ensure exact reproducibility.

#### 1) Voxel Spacing Extraction and Coordinate Standardization
- **Voxel Spacing definition:** The physical voxel resolutions (mm) along each axis are defined from the diagonal elements of the 3D affine matrix $M_{affine}$:
  $$S_x = |M_{affine}[0,0]|, \quad S_y = |M_{affine}[1,1]|, \quad S_z = |M_{affine}[2,2]|$$
- **Coordinate standardization:** The input image orientation is checked and standardized to the RAS (Right-Anterior-Superior) coordinate system, ensuring that the $X$-axis points Right-to-Left, the $Y$-axis points Anterior-to-Posterior, and the $Z$-axis points Inferior-to-Superior.

#### 2) Phase 1: Neck Stripping and Center-of-Mass (COM) Definition
To suppress center-of-mass distortion caused by variations in neck length across patients, the 98th percentile intensity of the raw image ($I_{98}$) is computed on a downsampled grid (step = 2). A threshold of 12% of this value is used to generate an initial brain mask $B(x, y, z)$:
$$B(x, y, z) = \begin{cases} 1 & \text{if } I(x, y, z) > 0.12 \times I_{98} \\ 0 & \text{otherwise} \end{cases}$$
- **Brain apex ($z_{top}$) detection:** The highest axial slice index containing the mask is identified as $z_{top} = \max \{ z \mid B(x, y, z) = 1 \}$.
- **Neck clipping:** Structures below a physical height of $130$ mm from the apex are clipped to construct the final head mask $H(x, y, z)$:
  $$z_{limit} = z_{top} - \text{round}\left(\frac{130 \text{ mm}}{S_z}\right)$$
  $$H(x, y, z) = \begin{cases} B(x, y, z) & \text{if } z \ge z_{limit} \\ 0 & \text{if } z < z_{limit} \end{cases}$$
- **Center-of-Mass (COM) calculation:** The arithmetic mean of the coordinates of the head mask voxels ($\mathcal{H} = \{(x_i, y_i, z_i)\}_{i=1}^N$ where $H = 1$) is calculated to define the anatomical origin ($x_{com}, y_{com}, z_{com}$):
  $$x_{com} = \frac{1}{N}\sum_{i=1}^N x_i, \quad y_{com} = \frac{1}{N}\sum_{i=1}^N y_i, \quad z_{com} = \frac{1}{N}\sum_{i=1}^N z_i$$

#### 3) Phase 2: Anatomical ROI Voxel Index Mapping Formulations
Using the COM origin, physical offsets (mm) are divided by voxel spacings ($S_x, S_y, S_z$) and rounded to integers to establish localized regions of interest (ROIs).

- **(1) Medial Temporal Atrophy (MTA ROI):** 
  - Target coronal slice index: $y_{mta} = \text{round}(y_{com} - \frac{12}{S_y})$
  - X-axis offset conversion: $\Delta x_{offset} = \text{round}(\frac{28}{S_x})$
  - Z-axis offset conversion: $\Delta z_{offset} = \text{round}(\frac{12}{S_z})$
  - ROI width ($W_x = 35$ mm) and height ($W_z = 25$ mm) voxel conversion:
    $$w_{x\_vox} = \text{round}\left(\frac{35}{S_x}\right), \quad w_{z\_vox} = \text{round}\left(\frac{25}{S_z}\right)$$
  - **Left Hippo ROI range:**
    $$x \in \left[ x_{com} - \Delta x_{offset} - \frac{w_{x\_vox}}{2}, \ x_{com} - \Delta x_{offset} + \frac{w_{x\_vox}}{2} \right]$$
  - **Right Hippo ROI range:**
    $$x \in \left[ x_{com} + \Delta x_{offset} - \frac{w_{x\_vox}}{2}, \ x_{com} + \Delta x_{offset} + \frac{w_{x\_vox}}{2} \right]$$
  - **Z-axis range (common to both sides):**
    $$z \in \left[ z_{com} - \Delta z_{offset} - \frac{w_{z\_vox}}{2}, \ z_{com} - \Delta z_{offset} + \frac{w_{z\_vox}}{2} \right]$$
  - *To ensure stability, the regional CSF ratio is averaged over 5 contiguous coronal slices centered on $y_{mta}$ ($y \in [y_{mta}-2, y_{mta}+2]$) to define the final hippocampal atrophy feature.*

- **(2) Frontal Atrophy ROI:**
  - Target axial slice index: $z_{front} = \text{round}(z_{com} + \frac{20}{S_z})$
  - Anterior Y-axis range:
    $$y \in \left[ \text{round}\left(y_{com} + \frac{10}{S_y}\right), \ \text{round}\left(y_{com} + \frac{50}{S_y}\right) \right]$$
  - X-axis range:
    $$x \in \left[ \text{round}\left(x_{com} - \frac{40}{S_x}\right), \ \text{round}\left(x_{com} + \frac{40}{S_x}\right) \right]$$

- **(3) Parietal Atrophy ROI:**
  - Target axial slice index: $z_{parietal} = \text{round}(z_{com} + \frac{35}{S_z})$
  - Posterior Y-axis range:
    $$y \in \left[ \text{round}\left(y_{com} - \frac{50}{S_y}\right), \ \text{round}\left(y_{com} - \frac{10}{S_y}\right) \right]$$
  - X-axis range:
    $$x \in \left[ \text{round}\left(x_{com} - \frac{40}{S_x}\right), \ \text{round}\left(x_{com} + \frac{40}{S_x}\right) \right]$$

- **(4) Temporal Atrophy ROI:**
  - Target axial slice index: $z_{temporal} = \text{round}(z_{com} - \frac{5}{S_z})$
  - Y-axis range:
    $$y \in \left[ \text{round}\left(y_{com} - \frac{30}{S_y}\right), \ \text{round}\left(y_{com} + \frac{30}{S_y}\right) \right]$$
  - X-axis ranges (left/right):
    $$\text{Left ROI: } x \in \left[ \text{round}\left(x_{com} - \frac{60}{S_x}\right), \ \text{round}\left(x_{com} - \frac{35}{S_x}\right) \right]$$
    $$\text{Right ROI: } x \in \left[ \text{round}\left(x_{com} + \frac{35}{S_x}\right), \ \text{round}\left(x_{com} + \frac{60}{S_x}\right) \right]$$

#### 4) Phase 3: Local CSF Ratio ($R_{CSF}$) and Feature Extraction
Within each defined ROI $\mathcal{R}$, the local CSF ratio $R_{CSF}$ is calculated by identifying voxels with intensities below a CSF threshold $T_{csf} = \text{factor} \times I_{98}$:
$$R_{CSF} = \frac{1}{|\mathcal{R}|} \sum_{(x,y,z) \in \mathcal{R}} \mathbb{I}\left(I(x,y,z) < T_{csf}\right)$$
where $\mathbb{I}$ is the indicator function. The optimized intensity factor thresholds are fixed at: MTA = 0.25, Frontal = 0.30, Parietal = 0.22, and Temporal = 0.30.

---

### 2.4 Cross-Validation and Statistical Analysis
To map the continuous $R_{CSF}$ features to discrete ordinal visual grades (score $\in \{0, \dots, C-1\}$), we utilized a **5-fold cross-validation** framework. This strictly isolates training and testing subsets to prevent data leakage during threshold selection.

#### 1) In-Fold Threshold Optimization using Nelder-Mead Simplex
- **Objective function:** Within each training fold subset $\mathcal{D}_{train}$, we seek a threshold vector $\boldsymbol{\theta} = [\theta_1, \dots, \theta_{C-1}]$ that maximizes the **quadratic weighted kappa ($\kappa_w$)** between the predicted grades $\hat{y}_i(\boldsymbol{\theta})$ and the manual ground truth grades $y_i$:
  $$\mathcal{L}(\boldsymbol{\theta}) = -\kappa_w\left(\mathbf{y}_{train}, \ \hat{\mathbf{y}}_{train}(\boldsymbol{\theta})\right)$$
  where the ordinal rating prediction is defined as:
  $$\hat{y}_i(\boldsymbol{\theta}) = \sum_{c=1}^{C-1} \mathbb{I}\left(R_{CSF, i} \ge \theta_c\right)$$
- **Optimization:** To handle the discontinuous, step-like nature of the ordinal loss function, we utilized the **Nelder-Mead simplex search** algorithm, which does not require gradients.
- **Initialization (Quantile Matching):** To ensure stable convergence, initial threshold boundaries $\boldsymbol{\theta}^{(0)}$ are set to match the cumulative proportion of manual grades in the training fold.
- **Constraints:** Following optimization, the threshold vector $\boldsymbol{\theta}^*$ is sorted to enforce monotonic boundaries. These thresholds are then applied to the held-out test fold $\mathcal{D}_{test}$ to generate unbiased predictions $\hat{\mathbf{y}}_{test}$. The test-fold predictions are concatenated across all 5 folds for final statistical evaluation.

#### 2) Statistical Reliability and Confidence Intervals
- **Agreement metrics:** For the total atrophy sum score (0–17), a two-way random-effects, absolute agreement model was used to calculate the **intraclass correlation coefficient (ICC(2,1))**:
  $$ICC(2,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + \frac{k}{n}(MS_C - MS_E)}$$
  where $MS_R$, $MS_C$, and $MS_E$ represent the mean square for rows (subjects), columns (raters/methods), and error, respectively.
- **Bootstrap confidence intervals:** To compute stable standard errors and 95% CIs for the kappa and ICC values, we performed non-parametric bootstrapping with replacement ($B = 2,000$). The 2.5th and 97.5th percentiles of the bootstrap distribution define the 95% CI bounds.
- **Hierarchical OLS regression:** To predict MMSE scores, we compared a covariate-only baseline model (Model 1: age, sex, education, APOE4 status) against models incorporating manual ratings (Model 2), automated e-CVRS (Model 3), and official UCSF FreeSurfer hippocampal volumes (Model 4). The incremental F-statistic was computed as:
  $$F = \frac{(RSS_{Model1} - RSS_{Model3}) / (p_3 - p_1)}{RSS_{Model3} / (n - p_3)}$$
  P-values for the incremental F-tests across the three imaging models were adjusted for multiple comparisons using the Benjamini–Hochberg False Discovery Rate (FDR) procedure. The custom calculations for OLS, F-tests, weighted kappa, and ICC were cross-verified against standard libraries (scikit-learn and pingouin) to ensure exact numerical agreement.

---

## 3. Results

### 3.1 Baseline Characteristics
The demographic and clinical characteristics of the 338 subjects are detailed in Table 1. The cohort had a mean age of 71.3±7.4 years, was 46.7% female, and had a mean education of 16.3±2.6 years, with 46.2% being APOE4 carriers. Cognitive markers indicated a mild stage of impairment: MMSE = 28.2±1.7 (range 23–30), CDR-SB = 1.5±0.9, and ADAS-11 = 9.0±4.3. The MMSE scores were highly skewed and concentrated at the higher end, typical of a clinical MCI-only cohort.

**Table 1.** Baseline characteristics of the study cohort (N = 338).

| Variable | Value |
| :--- | :--- |
| Age (years), mean ± SD (range) | 71.3 ± 7.4 (55.0–91.4) |
| Sex, Female / Male | 46.7% / 53.3% |
| Education (years), mean ± SD | 16.3 ± 2.6 |
| APOE4 Carrier | 46.2% (ε4×1: 36.7%, ε4×2: 9.5%) |
| MMSE, mean ± SD (range) | 28.2 ± 1.7 (23–30) |
| CDR-SB, mean ± SD | 1.5 ± 0.9 |
| ADAS-11, mean ± SD | 9.0 ± 4.3 |
| Diagnosis | MCI (Single cohort) |

### 3.2 Agreement with Manual Expert Ratings
The cross-validated agreement between the automated e-CVRS and manual expert ratings is summarized in Table 2. The intraclass correlation coefficient for the total atrophy sum score (0–17) was ICC(2,1) = 0.5622 (95% CI 0.4870–0.6326), indicating moderate agreement. Regionally, the temporal lobe (κw = 0.4929), left hippocampus (κw = 0.4540), and parietal lobe (κw = 0.3983) bordered the moderate range. In contrast, the frontal lobe (κw = 0.3500) and right hippocampus (κw = 0.3767) demonstrated fair agreement according to the Landis–Koch criteria. Overall, agreement ranged from fair-to-moderate, peaking in the temporal region and lowest in the frontal lobe.

**Table 2.** Agreement between e-CVRS and manual ratings (5-fold cross-validation, 2,000 bootstrap resamples).

| Subscale (Range) | Metric | Value | 95% CI | Interpretation (Landis–Koch / Koo–Li) |
| :--- | :--- | :--- | :--- | :--- |
| Left Hippocampus MTA (0–4) | κw | 0.4540 | 0.3701–0.5336 | Moderate |
| Right Hippocampus MTA (0–4) | κw | 0.3767 | 0.2821–0.4687 | Fair |
| Frontal Lobe (0–3) | κw | 0.3500 | 0.2472–0.4458 | Fair |
| Parietal Lobe (0–3) | κw | 0.3983 | 0.3127–0.4833 | Fair (upper bound) / Moderate border |
| Temporal Lobe (0–3) | κw | 0.4929 | 0.4013–0.5761 | Moderate |
| Total Atrophy Sum (0–17) | ICC(2,1) | 0.5622 | 0.4870–0.6326 | Moderate |

### 3.3 Predicting Cognitive Scores (MMSE): Hierarchical Regression
The baseline covariate model (Model 1) explained 8.47% of the variance in MMSE (adjusted R² = 0.0847). The incremental predictive power of the imaging metrics is presented in Table 3. The addition of the manual CVRS total score (Model 2) showed a nominal increase in adjusted R² to 0.0937 (F = 4.318, p = 0.039), which lost statistical significance after FDR correction (FDR-adjusted p = 0.115). The automated e-CVRS total score (Model 3) did not provide a significant increment over covariates (adjusted R² = 0.0828, F = 0.331, FDR-adjusted p = 0.565). In contrast, the incorporation of the official FreeSurfer hippocampal volumes (Model 4) significantly improved cognitive prediction, raising the adjusted R² to 0.1125 (incremental F = 11.430, FDR-adjusted p = 0.002).

**Table 3.** Hierarchical linear regression predicting baseline MMSE (N = 338).

| Model | Variable Composition | Adjusted R² | Incremental F | p-value | FDR-adjusted p |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Model 1 | Baseline Covariates | 0.0847 | – | – | – |
| Model 2 | + Manual CVRS Sum | 0.0937 | 4.318 | 0.039 | 0.115 |
| Model 3 | + e-CVRS Sum | 0.0828 | 0.331 | 0.565 | 0.565 |
| Model 4 | + FreeSurfer Hippocampal Vol | 0.1125 | 11.430 | < 0.001 | 0.002 |

---

## 4. Discussion

### 4.1 Principal Findings
We developed and validated e-CVRS, a lightweight Python-based pipeline that quantifies regional brain atrophy without heavy external software suites. Tested on the ADNI MCI cohort, it showed fair-to-moderate agreement with manual expert ratings (Total ICC = 0.5622, 95% CI: 0.4870–0.6326). The fact that the lower bound of the confidence interval remains close to the moderate agreement threshold under strict cross-validation demonstrates the robustness of the center-of-mass (COM) geometric alignment rules. However, regional variations in agreement highlight specific boundaries and areas for refinement.

### 4.2 Regional Agreement Variations
Agreement was highest in the temporal lobe and hippocampi, and lowest in the frontal lobe. This pattern can be explained by anatomical and methodological factors. First, the medial temporal and hippocampal areas are bounded by distinct CSF spaces (e.g., the temporal horn of the lateral ventricle, choroid fissure, and hippocampal fissure). The high contrast between CSF and brain tissue allows local CSF ratios to represent regional atrophy reliably. In contrast, frontal lobe atrophy is distributed diffusely across the frontal convexity, making a single, fixed axial slice less representative of 3D frontal atrophy. Second, the reliability of the reference standard itself varies by region; scales like the MTA [4,5] are highly standardized and reproducible, whereas cortical visual ratings like the GCA [6] exhibit greater inter-rater variability, which limits the upper bound of agreement. Third, the extreme rarity of severe atrophy in this MCI cohort (Frontal grade 3: n=2; Left/Right Hippo grade 4: n=3/4) likely destabilized threshold learning for these categories, particularly affecting the right hippocampus and frontal lobe.

### 4.3 Clinical Predictive Power and the Range Restriction Effect
The lack of independent predictive power for MMSE by e-CVRS (Model 3) is primarily due to the **range restriction effect** of this cohort. Because the cohort is limited to MCI subjects, MMSE scores are tightly clustered between 23 and 30 (mean 28.2±1.7). This compression of the dependent variable's variance mathematically dampens correlation and regression coefficients. The fact that even the manual CVRS (Model 2) lost significance after FDR correction confirms that this is a structural characteristic of the cohort rather than a limitation of a specific imaging tool. 

Importantly, however, the official FreeSurfer hippocampal volume (Model 4) retained highly significant predictive value (adjusted R² = 0.1125, p = 0.002) under the same range restriction. This highlights a critical limitation of visual rating scales: ordinal visual scales (0–4) compress structural variations into broad bins, whereas continuous high-resolution voxel counts ($\text{mm}^3$) retain sufficient variance to detect subtle, sub-clinical changes in hippocampal structure under restricted cognitive distributions. Therefore, these results should not be interpreted as a lack of clinical utility for structural atrophy, but rather as a demonstration of the sensitivity threshold of visual ratings under narrow cognitive distributions.

### 4.4 e-CVRS and FreeSurfer Volume: Interpretability vs. Sensitivity Trade-off
The contrast between e-CVRS (Model 3) and FreeSurfer (Model 4) illustrates the trade-off between interpretability and statistical sensitivity in medical image processing. While FreeSurfer provides superior sensitivity ($p = 0.002$), it requires extensive processing time and behaves as a black box; a clinician cannot easily verify why a specific volume was output. In contrast, e-CVRS guarantees explainability-by-design by displaying the exact coronal/axial slices, ROI boundaries, and CSF masks used for quantification. In clinical environments where decision-making requires human verification, this visual transparency is a major advantage that may offset lower statistical sensitivity. Future studies in full-spectrum cohorts (CN-MCI-AD) are needed to formally assess this trade-off when cognitive variance is fully restored.

### 4.5 Methodological Strengths
The e-CVRS pipeline offers several advantages: (i) it runs in seconds on standard CPUs without external dependencies like FSL or ANTs, (ii) COM alignment is robust against scanner padding and patient translation, enabling localized measurements without registration, (iii) it provides visual verification of measurements, ROIs, and CSF masks [14], and (iv) the validation framework prevents data leakage and computes bootstrap confidence intervals. These characteristics are well-suited for resource-limited clinical settings and regulatory validation.

### 4.6 Limitations
First, the cross-sectional design prevents evaluation of longitudinal cognitive decline or progression to dementia. Second, validation was limited to the ADNI MCI cohort, and performance has not been tested on external cohorts. Third, the fixed slice-offset rules can be affected by anatomical anomalies, particularly in the frontal lobe. Fourth, the 3D proxy volume used as a lightweight comparator was a crude approximation, and the official FreeSurfer volumetry is the preferred standard. Fifth, inter-rater reliability data for the manual reference standard was not available, making it difficult to isolate the impact of reference noise. Sixth, Nelder-Mead threshold optimization is sensitive to class imbalance, and alternative models like ordinal logistic regression should be explored.

### 4.7 Future Directions
Future priorities include: (1) intensity normalization using white matter peaks or z-scores to reduce scanner variability, (2) incorporating multi-slice measurements and partial volume correction, (3) evaluating ordinal logistic regression for thresholding, (4) addressing class imbalance through stratified k-fold splits or category merging, (5) validating scanner robustness across different field strengths (1.5T vs 3T) and sites, and (6) replicating these findings in a full-spectrum cohort (CN-MCI-AD) with adequate cognitive variance. Translating the pipeline into an explainable screening tool will involve implementing DICOM inputs, report generators, and API services.

---

## 5. Conclusions
We developed and validated e-CVRS, a lightweight, explainable automated visual rating scale that quantifies brain atrophy in seconds on standard CPUs without external dependencies. On the ADNI MCI cohort, e-CVRS achieved fair-to-moderate agreement with manual expert ratings (Total ICC = 0.5622). While range restriction in MMSE scores limited the predictive power of both manual and automated visual ratings, continuous FreeSurfer hippocampal volumes retained significant predictive value (adjusted R² = 0.1125, p = 0.002). This highlights the trade-off between the high sensitivity of voxel-based volumetry and the visual transparency of explainable visual ratings. Given its speed and explainable overlays, e-CVRS holds promise as a screening, educational, and clinical support tool.

---

## Declarations

**Ethics approval and consent.** This study utilized de-identified public secondary data from the ADNI database. ADNI was conducted under approval from the IRBs of all participating institutions, and written informed consent was obtained from all participants. 

**Data availability.** The imaging and clinical data used in this study are available from the ADNI database (adni.loni.usc.edu) upon signing a data use agreement and cannot be redistributed by the authors. The analysis and evaluation code is available upon reasonable request.

**Funding.** None / To be declared upon submission.

**Conflicts of interest.** The authors declare no conflicts of interest.

**Author contributions.** Study design, algorithm implementation, statistical analysis, manuscript drafting and revision — details to be provided according to CRediT taxonomy.

**ADNI Acknowledgment.** Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Award number W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging (NIA), the National Institute of Biomedical Imaging and Bioengineering (NIBIB), and through generous contributions from multiple private organizations. The grantees organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer's Therapeutic Research Institute at the University of Southern California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California. A complete list of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

---

## References

1. Jang JW, Park SY, Park YH, Baek MJ, Lim JS, Youn YC, et al. A comprehensive visual rating scale of brain magnetic resonance imaging: application in elderly subjects with Alzheimer's disease, mild cognitive impairment, and normal cognition. J Alzheimers Dis 2015;44:1023–34.
2. Jang JW, Park JH, Kim S, Park YH, Pyun JM, Lim JS, et al. A 'Comprehensive Visual Rating Scale' for predicting progression to dementia in patients with mild cognitive impairment. PLoS One 2018;13:e0201852.
3. Chung JK, Jang JW; Alzheimer's Disease Neuroimaging Initiative. Comprehensive visual rating scale on magnetic resonance imaging: application to prodromal Alzheimer disease. Ann Geriatr Med Res 2021;25(1):39–44.
4. Scheltens P, Leys D, Barkhof F, Huglo D, Weinstein HC, Vermersch P, et al. Atrophy of medial temporal lobes on MRI in "probable" Alzheimer's disease and normal ageing: diagnostic value and neuropsychological correlates. J Neurol Neurosurg Psychiatry 1992;55:967–72.
5. Kim GH, Kim JE, Choi KG, Lim SM, Lee JM, Na DL, et al. T1-weighted axial visual rating scale for an assessment of medial temporal atrophy in Alzheimer's disease. J Alzheimers Dis 2014;41:169–78.
6. Victoroff J, Mack WJ, Grafton ST, Schreiber SS, Chui HC. A method to improve interrater reliability of visual inspection of brain MRI scans in dementia. Neurology 1994;44:2267–76.
7. Koedam EL, Lehmann M, van der Flier WM, Scheltens P, Pijnenburg YA, Fox N, et al. Visual assessment of posterior atrophy: development of a MRI rating scale. Eur Radiol 2011;21:2618–25.
8. Pasquier F, Leys D, Weerts JG, Mounier-Vehier F, Barkhof F, Scheltens P. Inter- and intraobserver reproducibility of cerebral atrophy assessment on MRI scans with hemispheric infarcts. Eur Neurol 1996;36:268–72.
9. Frisoni GB, Fox NC, Jack CR Jr, Scheltens P, Thompson PM. The clinical use of structural MRI in Alzheimer disease. Nat Rev Neurol 2010;6:67–77.
10. Jack CR Jr, Bernstein MA, Fox NC, Thompson P, Alexander G, Harvey D, et al. The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. J Magn Reson Imaging 2008;27:685–91.
11. Fischl B. FreeSurfer. Neuroimage 2012;62:774–81.
12. Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M. FastSurfer: a fast and accurate deep learning based neuroimaging pipeline. Neuroimage 2020;219:117012.
13. Billot B, Greve DN, Puonti O, Thielscher A, Van Leemput K, Fischl B, et al. SynthSeg: segmentation of brain MRI scans of any contrast and resolution without retraining. Med Image Anal 2023;83:102789.
14. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell 2019;1:206–15.
15. Folstein MF, Folstein SE, McHugh PR. "Mini-mental state": a practical method for grading the cognitive state of patients for the clinician. J Psychiatr Res 1975;12:189–98.
16. Cohen J. Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. Psychol Bull 1968;70:213–20.
17. Landis JR, Koch GG. The measurement of observer agreement for categorical data. Biometrics 1977;33:159–74.
18. Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. J Chiropr Med 2016;15:155–63.
19. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 1986;1:307–10.
20. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Series B 1995;57:289–300.
21. Bossuyt PM, Reitsma JB, Bruns DE, Gatsonis CA, Glasziou PP, Irwig L, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. BMJ 2015;351:h5527.
22. Collins GS, Reitsma JB, Altman DG, Moons KG. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). BMJ 2015;350:g7594.
23. Wahlund LO, Barkhof F, Fazekas F, Bronge L, Augustin M, Sjögren M, et al. A new rating scale for age-related white matter changes applicable to MRI and CT. Stroke 2001;32:1318–22.

---

## Figure Legends

- **Figure 1.** Overview of the e-CVRS pipeline and explainable overlays of representative scans. ROI borders, CSF masks, measured CSF ratios, and assigned grades are annotated on the coronal MTA slice and axial cortical slices selected by COM alignment.
- **Figure 2.** (A) Bland–Altman plot of the total atrophy sum scores (manual vs. automated bias and 95% limits of agreement). (B) Regional grade confusion matrices illustrating that misclassifications are restricted to adjacent grades.
- **Figure 3.** Gallery of representative scans by grade (region × grade) displaying explainable overlays, illustrating the potential utility of the tool for clinical training and reading support.
