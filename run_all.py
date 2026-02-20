#!/usr/bin/env python3
"""
run_all.py
Usage:
  python run_all.py --input data.csv --outdir outputs
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "01_prevalence_rank.py", "--input", args.input, "--outdir", args.outdir])
    run([sys.executable, "02_associations_pairs.py", "--input", args.input, "--outdir", args.outdir, "--max_cols", "80"])
    run([sys.executable, "03_clustering_profiles.py", "--input", args.input, "--outdir", args.outdir, "--k", "5"])


if __name__ == "__main__":
    main()
