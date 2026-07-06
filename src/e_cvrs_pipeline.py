import sys
import os
import argparse

# Ensure E:\CVRS_MCI_APET_ADNI is in python path to resolve ecvrs package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecvrs.cli import main

if __name__ == "__main__":
    # Intercept arguments to bridge with modern ecvrs.cli subcommand structure
    parser = argparse.ArgumentParser(description="Legacy Wrapper for e-CVRS T1 MRI Features Extraction Pipeline")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing Analyze (.hdr) scans")
    parser.add_argument("--output_csv", type=str, default="e-CVRS_automated_scores.csv", help="CSV path for extracted features")
    
    parsed, unknown = parser.parse_known_args()
    
    new_args = [
        "cli.py", "extract",
        "--input_dir", parsed.input_dir,
        "--output_csv", parsed.output_csv,
        "--qc_csv", "qc_report.csv"
    ]
    new_args.extend(unknown)
    
    sys.argv = new_args
    sys.exit(main())
