import os
import math
import time
import random
import argparse
from typing import List, Dict, Set

import pandas as pd
import requests

ACS_YEAR = 2023
BASE = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
STATEFP_RI = "44"

# -----------------------------
# FULL ACS VARIABLE LIST (your spec)
# - C18108_001E excluded (often not available in /acs/acs5)
# - B26001_* included; if API rejects it, we skip and record in invalid vars file
# -----------------------------
ACS_VARS = sorted(set([
    # Population
    "B01003_001E",

    # Demographics
    "B01001_001E","B01001_002E",
    "B01001_007E","B01001_008E","B01001_009E","B01001_010E","B01001_011E",
    "B01001_031E","B01001_032E","B01001_033E","B01001_034E","B01001_035E",
    "B01001_012E","B01001_013E","B01001_014E","B01001_036E","B01001_037E","B01001_038E",
    "B01001_015E","B01001_016E","B01001_017E","B01001_018E","B01001_019E",
    "B01001_039E","B01001_040E","B01001_041E","B01001_042E","B01001_043E",
    "B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
    "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E",

    # Median age
    "B01002_001E",

    # Race/Ethnicity
    "B03002_001E","B03002_012E",
    "B03002_003E","B03002_004E","B03002_005E","B03002_006E","B03002_008E",

    # Income
    "B19013_001E",
    "B19301_001E",

    # Poverty
    "B17001_001E","B17001_002E",
    "B17001_003E","B17001_017E",
    "B17001_004E","B17001_005E","B17001_006E","B17001_007E","B17001_008E","B17001_009E","B17001_010E","B17001_011E",
    "B17001_018E","B17001_019E","B17001_020E","B17001_021E","B17001_022E","B17001_023E",
    "B17001_015E","B17001_016E","B17001_028E","B17001_029E","B17001_030E",

    # Inequality
    "B19083_001E",

    # Public assistance
    "B19057_001E","B19057_002E",

    # Employment
    "B23025_001E","B23025_002E","B23025_003E","B23025_005E","B23025_007E",

    # Education
    "B15003_001E", *[f"B15003_{i:03d}E" for i in range(2, 26)],

    # Household
    "B11001_001E","B11001_003E","B11001_006E","B11001_007E",
    "B25010_001E",

    # Marital
    "B12001_001E","B12001_002E","B12001_003E","B12001_009E","B12001_010E",

    # Housing
    "B25002_001E","B25002_002E","B25002_003E",
    "B25003_001E","B25003_002E","B25003_003E",
    "B25064_001E","B25077_001E",
    "B25014_001E","B25014_005E","B25014_006E","B25014_007E",

    # Rent burden
    "B25070_001E","B25070_007E","B25070_008E","B25070_009E","B25070_010E","B25070_011E",

    # Mobility
    "B07003_001E","B07003_004E","B07003_007E","B07003_010E","B07003_013E",

    # Migration
    "B05002_001E","B05002_013E",

    # Health access
    "B27001_001E","B27001_005E","B27001_008E","B27001_011E","B27001_014E","B27001_017E","B27001_020E","B27001_023E","B27001_026E","B27001_029E",

    # Veterans
    "B21001_001E","B21001_002E",

    # Group quarters (may be invalid; handled)
    "B26001_001E","B26001_002E",
]))


def chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def is_json_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "")
    return "json" in ctype.lower()


def census_get(session: requests.Session, params: Dict, max_retries: int = 4) -> List:
    last_err = None
    last_r = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(BASE, params=params, timeout=90)
            last_r = r
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                time.sleep(0.7 * attempt + random.random() * 0.3)
                continue

            if not is_json_response(r):
                last_err = "Non-JSON response"
                time.sleep(0.7 * attempt + random.random() * 0.3)
                continue

            return r.json()

        except requests.exceptions.JSONDecodeError:
            last_err = "JSONDecodeError"
            time.sleep(0.7 * attempt + random.random() * 0.3)
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(0.7 * attempt + random.random() * 0.3)

    body = (last_r.text if last_r is not None else "")
    url = (last_r.url if last_r is not None else "")
    raise requests.HTTPError(f"Failed Census request after retries. Last error: {last_err}\nURL: {url}\nBody:\n{body[:600]}")


def fetch_vars_for_ri(session: requests.Session, api_key: str, vars_list: List[str], invalid: Set[str]) -> pd.DataFrame:
    """
    Fetch all RI tracts for given vars_list (state:44, county:*)
    If 400 occurs, split vars until invalid vars isolated, then skip invalid vars.
    """
    def _try(vlist: List[str]) -> pd.DataFrame:
        params = {
            "get": ",".join(vlist),
            "for": "tract:*",
            "in": f"state:{STATEFP_RI} county:*",
            "key": api_key
        }
        data = census_get(session, params)
        df = pd.DataFrame(data[1:], columns=data[0])
        df["GEOID"] = df["state"].str.zfill(2) + df["county"].str.zfill(3) + df["tract"].str.zfill(6)
        return df.drop(columns=["state", "county", "tract"])

    try:
        return _try(vars_list)

    except requests.HTTPError:
        if len(vars_list) == 1:
            bad = vars_list[0]
            invalid.add(bad)
            return pd.DataFrame(columns=["GEOID", bad])

        mid = len(vars_list) // 2
        left = fetch_vars_for_ri(session, api_key, vars_list[:mid], invalid)
        right = fetch_vars_for_ri(session, api_key, vars_list[mid:], invalid)
        return left.merge(right, on="GEOID", how="outer")


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c != "GEOID":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_features(acs_raw: pd.DataFrame) -> pd.DataFrame:
    acs = to_numeric(acs_raw)
    out = pd.DataFrame({"GEOID": acs["GEOID"]})

    def col(name: str) -> pd.Series:
        return acs[name] if name in acs.columns else pd.Series([math.nan] * len(acs))

    # Population
    out["pop_total"] = col("B01003_001E")

    # Percent male
    out["pct_male"] = col("B01001_002E") / col("B01001_001E")

    # Age bands
    age_15_24 = acs[[c for c in [
        "B01001_007E","B01001_008E","B01001_009E","B01001_010E","B01001_011E",
        "B01001_031E","B01001_032E","B01001_033E","B01001_034E","B01001_035E"
    ] if c in acs.columns]].sum(axis=1)
    out["pct_age_15_24"] = age_15_24 / col("B01001_001E")

    age_25_44 = acs[[c for c in [
        "B01001_012E","B01001_013E","B01001_014E",
        "B01001_036E","B01001_037E","B01001_038E"
    ] if c in acs.columns]].sum(axis=1)
    out["pct_age_25_44"] = age_25_44 / col("B01001_001E")

    age_45_64 = acs[[c for c in [
        "B01001_015E","B01001_016E","B01001_017E","B01001_018E","B01001_019E",
        "B01001_039E","B01001_040E","B01001_041E","B01001_042E","B01001_043E"
    ] if c in acs.columns]].sum(axis=1)
    out["pct_age_45_64"] = age_45_64 / col("B01001_001E")

    age_65p = acs[[c for c in [
        "B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
        "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E"
    ] if c in acs.columns]].sum(axis=1)
    out["pct_age_65p"] = age_65p / col("B01001_001E")

    # Median age
    out["median_age"] = col("B01002_001E")

    # Race/Ethnicity
    out["pct_hispanic"] = col("B03002_012E") / col("B03002_001E")
    out["pct_nh_white"] = col("B03002_003E") / col("B03002_001E")
    out["pct_nh_black"] = col("B03002_004E") / col("B03002_001E")
    out["pct_nh_aian"]  = col("B03002_005E") / col("B03002_001E")
    out["pct_nh_asian"] = col("B03002_006E") / col("B03002_001E")
    out["pct_nh_two_or_more"] = col("B03002_008E") / col("B03002_001E")

    # Income
    out["median_hh_income"] = col("B19013_001E")
    out["per_capita_income"] = col("B19301_001E")

    # Poverty
    out["poverty_rate_all"] = col("B17001_002E") / col("B17001_001E")

    child_below = acs[[c for c in [
        "B17001_004E","B17001_005E","B17001_006E","B17001_007E","B17001_008E","B17001_009E","B17001_010E","B17001_011E",
        "B17001_018E","B17001_019E","B17001_020E","B17001_021E","B17001_022E","B17001_023E"
    ] if c in acs.columns]].sum(axis=1)
    child_total = col("B17001_003E") + col("B17001_017E")
    out["poverty_rate_child_u18"] = child_below / child_total

    elderly_below = acs[[c for c in ["B17001_015E","B17001_016E","B17001_028E","B17001_029E","B17001_030E"] if c in acs.columns]].sum(axis=1)
    out["poverty_rate_elderly_65p"] = elderly_below / age_65p

    # Gini
    out["gini_index"] = col("B19083_001E")

    # Public assistance
    out["hh_public_assistance_share"] = col("B19057_002E") / col("B19057_001E")

    # Employment
    out["lfp_rate_16p"] = col("B23025_002E") / col("B23025_001E")
    out["unemployment_rate"] = col("B23025_005E") / col("B23025_003E")
    out["not_in_labor_force_share_16p"] = col("B23025_007E") / col("B23025_001E")

    # Education
    lt_hs_cols = [f"B15003_{i:03d}E" for i in range(2, 17)]
    out["edu_lt_hs_25p"] = acs[[c for c in lt_hs_cols if c in acs.columns]].sum(axis=1) / col("B15003_001E")
    out["edu_hs_grad_25p"] = col("B15003_017E") / col("B15003_001E")
    out["edu_some_college_25p"] = acs[[c for c in ["B15003_018E","B15003_019E","B15003_020E"] if c in acs.columns]].sum(axis=1) / col("B15003_001E")
    out["edu_ba_plus_25p"] = acs[[c for c in ["B15003_022E","B15003_023E","B15003_024E","B15003_025E"] if c in acs.columns]].sum(axis=1) / col("B15003_001E")
    out["edu_grad_prof_25p"] = acs[[c for c in ["B15003_023E","B15003_024E","B15003_025E"] if c in acs.columns]].sum(axis=1) / col("B15003_001E")

    # Household
    out["hh_total"] = col("B11001_001E")
    out["hh_single_person_share"] = col("B11001_007E") / col("B11001_001E")
    out["hh_married_couple_share"] = col("B11001_003E") / col("B11001_001E")
    out["hh_female_householder_no_spouse_share"] = col("B11001_006E") / col("B11001_001E")
    out["avg_hh_size"] = col("B25010_001E")

    # Marital
    out["marital_never_married_share_15p"] = col("B12001_003E") / col("B12001_001E")
    out["marital_divorced_share_15p"] = col("B12001_009E") / col("B12001_001E")
    out["marital_separated_share_15p"] = col("B12001_010E") / col("B12001_001E")
    out["marital_married_share_15p"] = col("B12001_002E") / col("B12001_001E")

    # Housing
    out["housing_occupied_share"] = col("B25002_002E") / col("B25002_001E")
    out["housing_vacancy_rate"] = col("B25002_003E") / col("B25002_001E")
    out["homeownership_rate"] = col("B25003_002E") / col("B25003_001E")
    out["renter_occupied_share"] = col("B25003_003E") / col("B25003_001E")
    out["median_gross_rent"] = col("B25064_001E")
    out["median_home_value"] = col("B25077_001E")

    overcrowded = acs[[c for c in ["B25014_005E","B25014_006E","B25014_007E"] if c in acs.columns]].sum(axis=1)
    out["overcrowding_rate_gt1pproom"] = overcrowded / col("B25014_001E")

    rent_burden_num = acs[[c for c in ["B25070_007E","B25070_008E","B25070_009E","B25070_010E"] if c in acs.columns]].sum(axis=1)
    rent_burden_den = (col("B25070_001E") - col("B25070_011E"))
    out["rent_burdened_gt30"] = rent_burden_num / rent_burden_den
    out["rent_burdened_gt50"] = col("B25070_010E") / rent_burden_den

    moved = acs[[c for c in ["B07003_004E","B07003_007E","B07003_010E","B07003_013E"] if c in acs.columns]].sum(axis=1)
    out["residential_stability_same_house_1yr"] = 1.0 - (moved / col("B07003_001E"))

    # Migration
    out["foreign_born_share"] = col("B05002_013E") / col("B05002_001E")

    # Uninsured
    uninsured_num = acs[[c for c in [
        "B27001_005E","B27001_008E","B27001_011E","B27001_014E","B27001_017E",
        "B27001_020E","B27001_023E","B27001_026E","B27001_029E"
    ] if c in acs.columns]].sum(axis=1)
    out["uninsured_rate"] = uninsured_num / col("B27001_001E")

    # Veterans
    out["veteran_share_18p"] = col("B21001_002E") / col("B21001_001E")

    # Group quarters (may be NaN)
    out["group_quarters_share"] = col("B26001_002E") / col("B26001_001E")

    return out


def save_per_geoid_tables(df: pd.DataFrame, outdir: str, prefix: str):
    """
    Saves one CSV per GEOID, each containing a single row.
    """
    os.makedirs(outdir, exist_ok=True)
    for _, row in df.iterrows():
        geoid = row["GEOID"]
        path = os.path.join(outdir, f"{prefix}_{geoid}.csv")
        row.to_frame().T.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True, help="Census API key")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--vars-per-call", type=int, default=25, help="Chunk size for 'get=' vars (default 25)")
    ap.add_argument("--save-per-geoid", action="store_true", help="If set, writes one CSV per GEOID (many files)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    invalid_vars: Set[str] = set()
    var_chunks = chunks(ACS_VARS, args.vars_per_call)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "ri-acs-fetch/1.0"})
        frames: List[pd.DataFrame] = []

        for vc in var_chunks:
            frames.append(fetch_vars_for_ri(session, args.api_key, vc, invalid_vars))

        # Merge all chunks horizontally on GEOID
        acs_raw = frames[0]
        for f in frames[1:]:
            acs_raw = acs_raw.merge(f, on="GEOID", how="outer")

    # Save raw table
    raw_path = os.path.join(args.outdir, f"ri_tract_acs_raw_{ACS_YEAR}.csv")
    acs_raw.to_csv(raw_path, index=False)
    print("Saved:", raw_path)

    # Derived features table
    features = compute_features(acs_raw)
    feat_path = os.path.join(args.outdir, f"ri_tract_acs_features_{ACS_YEAR}.csv")
    features.to_csv(feat_path, index=False)
    print("Saved:", feat_path)

    # Invalid vars list
    inv_path = os.path.join(args.outdir, "invalid_acs_vars.txt")
    with open(inv_path, "w", encoding="utf-8") as f:
        for v in sorted(invalid_vars):
            f.write(v + "\n")
    print("Saved:", inv_path)

    # Optional per-GEOID single-row tables (raw + features)
    if args.save_per_geoid:
        save_per_geoid_tables(
            acs_raw,
            outdir=os.path.join(args.outdir, "per_geoid_raw"),
            prefix=f"ri_acs_raw_{ACS_YEAR}"
        )
        print("Saved per-GEOID raw tables to:", os.path.join(args.outdir, "per_geoid_raw"))

        save_per_geoid_tables(
            features,
            outdir=os.path.join(args.outdir, "per_geoid_features"),
            prefix=f"ri_acs_features_{ACS_YEAR}"
        )
        print("Saved per-GEOID feature tables to:", os.path.join(args.outdir, "per_geoid_features"))


if __name__ == "__main__":
    main()