#!/usr/bin/env bash
# End-to-end reproduction of the published e-CVRS analysis (ADNI MCI, N=339).
#
# Prerequisites
#   1. ADNI access (https://adni.loni.usc.edu) and the screening T1 MRI set.
#   2. pip install -r requirements.txt
#   3. The ADNI-derived tables in the current directory (or ECVRS_DATA_DIR):
#      ADNI_MRI_rating.xlsx, UCSF_FS.csv, UWNPSYCHSUM_*.csv, UPENNBIOMK9_*.csv
#
# Usage
#   ./reproduce.sh /path/to/adni/scans
#
# Expected headline result (seed 42):
#   Total Sum ICC(2,1) = 0.7482 [95% CI 0.6961 - 0.7930]
#   Weighted kappa: Hippo_Lt 0.599, Hippo_Rt 0.622, Front 0.498,
#                   Parietal 0.594, Temporal 0.666
#   FreeSurfer hippocampal volume rho = 0.416; CSF Abeta42 rho = -0.282
set -euo pipefail

SCAN_DIR="${1:-}"
if [[ -z "$SCAN_DIR" ]]; then
  echo "usage: $0 /path/to/adni/scans" >&2
  exit 1
fi

export ECVRS_DATA_DIR="${ECVRS_DATA_DIR:-$PWD}"
export ECVRS_OUT_DIR="${ECVRS_OUT_DIR:-analysis_out}"
export ECVRS_FIG_DIR="${ECVRS_FIG_DIR:-figures_out}"

echo "==> Step 1/5: extracting transparent features from MRI"
python -m ecvrs.cli extract \
  --input_dir "$SCAN_DIR" \
  --output_csv e-CVRS_automated_scores.csv \
  --qc_csv qc_report.csv

echo "==> Step 2/5: quadratic-ridge calibration + agreement report"
python -m ecvrs.cli evaluate \
  --scores_csv e-CVRS_automated_scores.csv \
  --ratings_excel ADNI_MRI_rating.xlsx \
  --freesurfer_csv UCSF_FS.csv \
  --calibration quadratic_ridge \
  --seed 42 \
  --output_dir results

echo "==> Step 3/5: building the multi-domain cohort"
python analysis/merge_cohort.py

echo "==> Step 4/5: calibration-candidate benchmark (5 candidates x 8 seeds)"
python analysis/benchmark.py

echo "==> Step 5/5: main-text tables and figures"
python analysis/final_analysis.py
python analysis/make_figures.py

echo "==> Done."
echo "    agreement report : results/evaluation_report.txt"
echo "    tables (JSON)    : $ECVRS_OUT_DIR/final_results.json"
echo "    figures 1-4      : $ECVRS_FIG_DIR/"
echo "    figure 5 overlay : python -m ecvrs.cli render --scan <subject.hdr> --output_dir case_out"
