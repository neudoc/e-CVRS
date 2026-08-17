#!/usr/bin/env bash
# End-to-end reproduction of the published e-CVRS v3 analysis (ADNI MCI, N=339).
#
# Prerequisites
#   1. ADNI access (https://adni.loni.usc.edu) and the screening T1 MRI set.
#   2. pip install -r requirements.txt
#
# Usage
#   ./reproduce.sh /path/to/adni/scans
#
# Expected headline result (seed 42):
#   Total Sum ICC(2,1) = 0.7467 [95% CI 0.6942 - 0.7884]
#   Weighted kappa: Hippo_Lt 0.600, Hippo_Rt 0.622, Front 0.470,
#                   Parietal 0.631, Temporal 0.652
set -euo pipefail

SCAN_DIR="${1:-}"
if [[ -z "$SCAN_DIR" ]]; then
  echo "usage: $0 /path/to/adni/scans" >&2
  exit 1
fi

echo "==> Step 1/2: extracting transparent features from MRI"
python -m ecvrs.cli extract \
  --input_dir "$SCAN_DIR" \
  --output_csv e-CVRS_automated_scores.csv \
  --qc_csv qc_report.csv

echo "==> Step 2/2: adaptive calibration + statistical validation"
python -m ecvrs.cli evaluate \
  --scores_csv e-CVRS_automated_scores.csv \
  --ratings_excel ADNI_MRI_rating.xlsx \
  --freesurfer_csv UCSF_FS.csv \
  --calibration adaptive_ridge \
  --seed 42 \
  --output_dir results

echo "==> Done. See results/evaluation_report.txt"
