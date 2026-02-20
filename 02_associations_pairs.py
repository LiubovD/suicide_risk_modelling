#!/usr/bin/env python3
"""
02_associations_pairs.py
Purpose: Within suicide-only cases, find binary factor pairs that co-occur.
Outputs: association_pairs.csv, association_pairs.xlsx
Usage:
  python 02_associations_pairs.py --input data.csv --outdir outputs --target suicidedeath --max_cols 80
Notes:
  - Automatically tries to detect binary 0/1 columns (including Yes/No strings).
  - Use --max_cols to limit computation if you have many flags.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def coerce_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "true": 1, "false": 0,
        "t": 1, "f": 0,
        "unknown": np.nan, "nan": np.nan, "none": np.nan, "": np.nan
    }
    return s.map(mapping)


def phi_from_table(tab: np.ndarray) -> float:
    # tab is 2x2
    a, b, c, d = tab[0, 0], tab[0, 1], tab[1, 0], tab[1, 1]
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return np.nan
    return (a * d - b * c) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--target", default="suicidedeath")
    ap.add_argument("--min_non_missing", type=float, default=0.5)
    ap.add_argument("--max_cols", type=int, default=100,
                    help="Limit number of binary columns analyzed (top by prevalence).")
    ap.add_argument("--min_prevalence", type=float, default=0.05,
                    help="Minimum prevalence of '1' to include a binary column (0-1).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found.")

    y = pd.to_numeric(df[args.target], errors="coerce")
    df_s = df.loc[y == 1].copy()
    if df_s.empty:
        raise SystemExit("No suicide-only rows found (target != 1).")

    # Detect binary columns
    candidates = []
    for c in df_s.columns:
        if c == args.target:
            continue
        s = coerce_binary(df_s[c])
        non_missing_frac = 1.0 - s.isna().mean()
        if non_missing_frac < args.min_non_missing:
            continue
        uniq = set(pd.unique(s.dropna()))
        if len(uniq) > 0 and uniq.issubset({0, 1}):
            prev = float(np.nanmean(s.values))
            if prev >= args.min_prevalence and prev <= (1 - args.min_prevalence):
                candidates.append((c, prev))

    if not candidates:
        raise SystemExit("No suitable binary columns found (0/1 or yes/no).")

    # Keep top max_cols by prevalence (mid-range often best, but we just pick by prevalence)
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in candidates[: args.max_cols]]

    X = pd.DataFrame({c: coerce_binary(df_s[c]) for c in selected})
    # Drop rows with any missing in selected for pairwise exactness (alternative: pairwise drop)
    # We'll do pairwise drop to keep more data.

    rows = []
    for c1, c2 in combinations(selected, 2):
        s1 = X[c1]
        s2 = X[c2]
        mask = s1.notna() & s2.notna()
        if mask.sum() < 50:
            continue

        tab = pd.crosstab(s1[mask], s2[mask]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        if tab.shape != (2, 2):
            continue

        phi = phi_from_table(tab)
        try:
            chi2, p, _, _ = chi2_contingency(tab)
        except Exception:
            p = np.nan
            chi2 = np.nan

        rows.append({
            "var1": c1,
            "var2": c2,
            "n_pairwise": int(mask.sum()),
            "phi": float(phi) if phi is not None else np.nan,
            "chi2": float(chi2) if chi2 is not None else np.nan,
            "p_value": float(p) if p is not None else np.nan,
            "table_00": int(tab[0, 0]),
            "table_01": int(tab[0, 1]),
            "table_10": int(tab[1, 0]),
            "table_11": int(tab[1, 1]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No pairs met minimum data thresholds.")

    # Rank by absolute association strength
    out["abs_phi"] = out["phi"].abs()
    out = out.sort_values(["abs_phi", "n_pairwise"], ascending=False)

    out.to_csv(outdir / "association_pairs.csv", index=False)
    with pd.ExcelWriter(outdir / "association_pairs.xlsx") as w:
        out.to_excel(w, sheet_name="pairs_ranked", index=False)

    print(f"\nSuicide-only rows: {len(df_s)}")
    print(f"Binary columns analyzed: {len(selected)}")
    print("\nTop 20 associated pairs (by |phi|):")
    print(out[["var1", "var2", "phi", "p_value", "n_pairwise"]].head(20).to_string(index=False))
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
