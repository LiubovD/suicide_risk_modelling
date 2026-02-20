#!/usr/bin/env python3
"""
03_clustering_profiles.py
Purpose: Cluster suicide-only cases into profiles and summarize each cluster.
Outputs:
  clustered_cases.csv
  cluster_summary.csv
  cluster_summary.xlsx
Usage:
  python 03_clustering_profiles.py --input data.csv --outdir outputs --target suicidedeath --k 5
Notes:
  - Uses binary flags + one-hot for selected categoricals.
  - You should edit INCLUDED_CATEGORICALS to match your data and goals.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Edit this list (keep it small-ish). These create interpretable clusters.
INCLUDED_CATEGORICALS = [
    "Sex", "agecat", "raceeth", "RaceEthnicity_c", "marital", "education",
    "WeaponType1", "firearmcat", "ResidenceState", "InjuryState"
]


def detect_binary_columns(df: pd.DataFrame, exclude: set) -> list[str]:
    bin_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        # Numeric 0/1
        if pd.api.types.is_numeric_dtype(s):
            u = set(pd.unique(pd.to_numeric(s, errors="coerce").dropna()))
            if u and u.issubset({0, 1}):
                bin_cols.append(c)
        else:
            # Common yes/no strings
            sl = s.astype(str).str.strip().str.lower()
            u = set(pd.unique(sl.dropna()))
            if u and u.issubset({"0", "1", "yes", "no", "y", "n", "true", "false", "t", "f"}):
                bin_cols.append(c)
    return bin_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--target", default="suicidedeath")
    ap.add_argument("--k", type=int, default=5, help="Number of clusters")
    ap.add_argument("--random_state", type=int, default=42)
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

    exclude = {args.target, "PersonID", "IncidentNumber", "IncidentYear", "timeperiod"}
    binary_cols = detect_binary_columns(df_s, exclude=exclude)

    # Keep categoricals that exist
    categorical_cols = [c for c in INCLUDED_CATEGORICALS if c in df_s.columns and c not in exclude]
    # Numeric (non-binary) columns (optional; can add Age if desired)
    numeric_cols = []
    if "Age" in df_s.columns:
        numeric_cols.append("Age")

    # Build preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("bin", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
            ]), binary_cols),
            ("cat", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
            ("num", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = KMeans(n_clusters=args.k, random_state=args.random_state, n_init="auto")

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("kmeans", model),
    ])

    clusters = pipe.fit_predict(df_s)

    df_out = df_s.copy()
    df_out["cluster"] = clusters
    df_out.to_csv(outdir / "clustered_cases.csv", index=False)

    # Cluster summaries: prevalence of binary flags + key categoricals
    summaries = []
    for cl in sorted(df_out["cluster"].unique()):
        sub = df_out[df_out["cluster"] == cl]
        row = {"cluster": int(cl), "n": int(len(sub))}

        # Binary flag prevalence (as numeric mean)
        for c in binary_cols:
            s = pd.to_numeric(sub[c], errors="coerce")
            if s.dropna().isin([0, 1]).all() and s.notna().sum() > 0:
                row[f"{c}_prev"] = float(s.mean())

        # Top category for selected categoricals
        for c in categorical_cols:
            vc = sub[c].astype("string").fillna("<<MISSING>>").value_counts()
            row[f"{c}_top"] = str(vc.index[0])
            row[f"{c}_top_pct"] = float(vc.iloc[0]) / len(sub)

        # Numeric
        for c in numeric_cols:
            s = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(s.mean())
            row[f"{c}_median"] = float(s.median())

        summaries.append(row)

    summary_df = pd.DataFrame(summaries).sort_values("n", ascending=False)
    summary_df.to_csv(outdir / "cluster_summary.csv", index=False)
    with pd.ExcelWriter(outdir / "cluster_summary.xlsx") as w:
        summary_df.to_excel(w, sheet_name="cluster_summary", index=False)

    print(f"\nSuicide-only rows: {len(df_s)}")
    print(f"Binary cols used: {len(binary_cols)}")
    print(f"Categorical cols used: {len(categorical_cols)} -> {categorical_cols}")
    print(f"Numeric cols used: {numeric_cols}")
    print("\nCluster sizes:")
    print(summary_df[["cluster", "n"]].to_string(index=False))
    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Tip: Open cluster_summary.xlsx and look at high *_prev values per cluster to label profiles.")


if __name__ == "__main__":
    main()
