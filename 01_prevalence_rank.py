#!/usr/bin/env python3
"""
01_prevalence_rank.py
Purpose: Suicide-only prevalence ranking (most common factors among suicide deaths).
Outputs: prevalence_rank.csv, prevalence_rank.xlsx
Usage:
  python 01_prevalence_rank.py --input data.csv --outdir outputs --target suicidedeath
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def coerce_binary(series: pd.Series) -> pd.Series:
    """Try to coerce common yes/no encodings into 0/1 with NaN allowed."""
    if pd.api.types.is_numeric_dtype(series):
        # If it's numeric, keep it but normalize common encodings (e.g., 1/2 -> 0/1 is unknown)
        return series
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "true": 1, "false": 0,
        "t": 1, "f": 0,
        "positive": 1, "negative": 0,
        "unknown": np.nan, "nan": np.nan, "none": np.nan, "": np.nan
    }
    return s.map(mapping)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--target", default="suicidedeath", help="Target column (1=suicide)")
    ap.add_argument("--min_non_missing", type=float, default=0.3,
                    help="Minimum fraction of non-missing required to include a column (0-1)")
    ap.add_argument("--topn", type=int, default=50, help="How many top columns to show in console")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)

    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in columns.")

    # Filter to suicide-only cases
    # Coerce target to numeric if needed
    y = pd.to_numeric(df[args.target], errors="coerce")
    df_s = df.loc[y == 1].copy()

    if df_s.empty:
        raise SystemExit("No suicide cases found (target != 1). Check coding in suicidedeath.")

    # Drop obvious IDs (customize if you want)
    drop_like = {"PersonID", "IncidentNumber", "IncidentYear", "timeperiod"}
    cols = [c for c in df_s.columns if c not in drop_like and c != args.target]

    results = []
    n = len(df_s)

    for c in cols:
        s = df_s[c]
        non_missing_frac = 1.0 - (s.isna().mean())
        if non_missing_frac < args.min_non_missing:
            continue

        # Try to treat as binary if it looks binary after coercion
        s_bin = coerce_binary(s)
        # Determine if it behaves like 0/1
        unique_vals = pd.unique(s_bin.dropna())
        if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1}):
            prevalence = float(np.nanmean(s_bin.values))  # mean of 0/1 = prevalence
            results.append({
                "variable": c,
                "type": "binary_flag",
                "n": n,
                "non_missing": int(s_bin.notna().sum()),
                "prevalence_yes": prevalence,
                "prevalence_yes_pct": 100.0 * prevalence
            })
        else:
            # For categorical / text: show most common category and its pct
            vc = s.astype("string").fillna("<<MISSING>>").value_counts(dropna=False)
            top_val = str(vc.index[0])
            top_pct = float(vc.iloc[0]) / n
            results.append({
                "variable": c,
                "type": "categorical_or_numeric",
                "n": n,
                "non_missing": int(s.notna().sum()),
                "top_value": top_val,
                "top_value_pct": 100.0 * top_pct,
                "num_unique_non_missing": int(s.dropna().nunique())
            })

    out = pd.DataFrame(results)

    # Rank binary flags by prevalence of "yes"
    binary = out[out["type"] == "binary_flag"].copy()
    binary = binary.sort_values(["prevalence_yes"], ascending=False)

    # Rank categoricals by top-category dominance (optional)
    cat = out[out["type"] != "binary_flag"].copy()
    if "top_value_pct" in cat.columns:
        cat = cat.sort_values(["top_value_pct"], ascending=False)

    # Save
    binary.to_csv(outdir / "prevalence_rank_binary.csv", index=False)
    cat.to_csv(outdir / "prevalence_rank_categorical.csv", index=False)

    with pd.ExcelWriter(outdir / "prevalence_rank.xlsx") as w:
        binary.to_excel(w, sheet_name="binary_flags_ranked", index=False)
        cat.to_excel(w, sheet_name="categorical_ranked", index=False)

    print(f"\nSuicide-only rows: {n}")
    print("\nTop binary flags by prevalence (YES):")
    if not binary.empty:
        show = binary[["variable", "prevalence_yes_pct", "non_missing"]].head(args.topn)
        print(show.to_string(index=False))
    else:
        print("No binary-like columns detected. Check encodings.")

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
