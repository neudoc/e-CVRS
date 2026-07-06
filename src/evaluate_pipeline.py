import sys
import os
import argparse

# Ensure E:\CVRS_MCI_APET_ADNI is in python path to resolve ecvrs package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecvrs.cli import main

if __name__ == "__main__":
    # Intercept arguments to bridge with modern ecvrs.cli evaluation subcommand structure
    parser = argparse.ArgumentParser(description="Legacy Wrapper for e-CVRS Validation and Statistical Analysis Script")
    parser.add_argument("--scores_csv", type=str, default="e-CVRS_automated_scores.csv", help="Path to automated scores CSV")
    parser.add_argument("--ratings_excel", type=str, default="ADNI_MRI_rating.xlsx", help="Path to manual ratings Excel")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save statistical report and plots")
    
    parsed, unknown = parser.parse_known_args()
    
    # We default the calibration to ordinal (Nelder-Mead optimization) to reproduce the paper exactly
    new_args = [
        "cli.py", "evaluate",
        "--scores_csv", parsed.scores_csv,
        "--ratings_excel", parsed.ratings_excel,
        "--output_dir", parsed.output_dir,
        "--calibration", "ordinal",
        "--seed", "42"
    ]
    new_args.extend(unknown)
    
    sys.argv = new_args
    sys.exit(main())
