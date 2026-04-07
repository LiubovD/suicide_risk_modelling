"""
Add tract area + population density to training_table_bn.csv using ri_tract.geojson

Inputs:
- ri_tract.geojson   (tract geometries; must contain a GEOID column)
- training_table_bn.csv (must contain GEOID and pop_total)

Outputs:
- training_table_with_density.csv  (original table + area_km2, area_sqmi, pop_density, log_pop_density)
- tract_areas_from_geojson.csv     (GEOID + areas)

Notes:
- Area must be computed in a projected CRS (meters). For Rhode Island, EPSG:26919 (UTM 19N) is a good default.
- If your GeoJSON is already projected, the code will still reproject safely.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import geopandas as gpd


# -------------------------
# User settings
# -------------------------
GEOJSON_PATH = "ri_tract.geojson"
TRAINING_CSV_PATH = "training_table_bn.csv"
OUTPUT_CSV_PATH = "training_table_with_density.csv"
AREAS_ONLY_CSV_PATH = "tract_areas_from_geojson.csv"

# Rhode Island: UTM zone 19N (meters)
PROJECTED_EPSG = 26919

# If your GeoJSON uses a different GEOID field name, add it here
POSSIBLE_GEOID_COLS = ["GEOID", "geoid", "GEOID10", "GEOID20", "TRACTCE", "TRACTCE20"]


# -------------------------
# Helpers
# -------------------------
def find_geoid_column(gdf: gpd.GeoDataFrame) -> str:
    cols = list(gdf.columns)
    for c in POSSIBLE_GEOID_COLS:
        if c in cols:
            return c
    # Try case-insensitive match
    lower_map = {c.lower(): c for c in cols}
    for c in POSSIBLE_GEOID_COLS:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(
        f"Could not find a GEOID column in GeoJSON. Columns found: {cols}\n"
        f"Expected one of: {POSSIBLE_GEOID_COLS}"
    )


def normalize_geoid(series: pd.Series) -> pd.Series:
    # Ensure strings, strip whitespace, keep leading zeros (critical)
    s = series.astype(str).str.strip()
    return s


# -------------------------
# 1) Load tract geometries
# -------------------------
tracts = gpd.read_file(GEOJSON_PATH)

if tracts.empty:
    raise ValueError(f"No features found in {GEOJSON_PATH}")

geoid_col = find_geoid_column(tracts)
tracts = tracts.rename(columns={geoid_col: "GEOID"})
tracts["GEOID"] = normalize_geoid(tracts["GEOID"])

# Ensure we have geometries
if tracts.geometry.isna().any():
    bad = tracts[tracts.geometry.isna()][["GEOID"]].head(10)
    raise ValueError(f"Some features have missing geometry. Example GEOIDs:\n{bad}")

# -------------------------
# 2) Reproject to meters and compute area
# -------------------------
# If CRS missing, assume WGS84 (common for GeoJSON). Better to set correctly if you know it.
if tracts.crs is None:
    print("WARNING: GeoJSON has no CRS info. Assuming EPSG:4326 (WGS84 lat/lon).")
    tracts = tracts.set_crs(epsg=4326)

tracts_proj = tracts.to_crs(epsg=PROJECTED_EPSG)

# Compute area from geometry (m^2), then convert
area_m2 = tracts_proj.geometry.area
tracts_proj["area_km2"] = area_m2 / 1_000_000.0
tracts_proj["area_sqmi"] = area_m2 / 2_589_988.110336  # exact m^2 per sq mile

# Keep only what we need for merging
areas = tracts_proj[["GEOID", "area_km2", "area_sqmi"]].copy()

# Basic sanity checks
if (areas["area_km2"] <= 0).any():
    bad = areas[areas["area_km2"] <= 0].head(10)
    raise ValueError(f"Non-positive areas found (check projection/geometry). Example rows:\n{bad}")

# Save areas-only file (useful for QA)
areas.to_csv(AREAS_ONLY_CSV_PATH, index=False)
print(f"Saved areas table: {AREAS_ONLY_CSV_PATH}")

# -------------------------
# 3) Load training table and merge
# -------------------------
df = pd.read_csv(TRAINING_CSV_PATH, dtype={"GEOID": str})

required_cols = {"GEOID", "pop_total"}
missing_req = required_cols - set(df.columns)
if missing_req:
    raise ValueError(f"training_table_bn.csv is missing required columns: {sorted(missing_req)}")

df["GEOID"] = normalize_geoid(df["GEOID"])

df_merged = df.merge(areas, on="GEOID", how="left", validate="m:1")

# Check for unmatched GEOIDs
n_missing_area = df_merged["area_km2"].isna().sum()
if n_missing_area > 0:
    missing_geoids = df_merged.loc[df_merged["area_km2"].isna(), "GEOID"].unique().tolist()
    print(f"WARNING: {n_missing_area} rows missing area after merge.")
    print("Example missing GEOIDs:", missing_geoids[:20])
    print("Fix by ensuring the same vintage/year GEOIDs and same GEOID formatting.")
else:
    print("All GEOIDs matched to areas successfully.")

# -------------------------
# 4) Compute population density
# -------------------------
df_merged["pop_total"] = pd.to_numeric(df_merged["pop_total"], errors="coerce")

if df_merged["pop_total"].isna().any():
    bad = df_merged[df_merged["pop_total"].isna()][["GEOID", "pop_total"]].head(10)
    raise ValueError(f"Found non-numeric pop_total values. Example rows:\n{bad}")

df_merged["pop_density"] = df_merged["pop_total"] / df_merged["area_km2"]

# log density: handle zeros safely (shouldn't happen if pop_total > 0)
df_merged["log_pop_density"] = np.log(df_merged["pop_density"].replace({0: np.nan}))

# Optional: if any pop_total is 0, log becomes NaN. You can decide how to handle:
# df_merged["log_pop_density"] = np.log1p(df_merged["pop_density"])  # alternative

# -------------------------
# 5) Save updated table
# -------------------------
df_merged.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Saved updated training table: {OUTPUT_CSV_PATH}")

# Quick summary for QA
print("\n--- QA summary ---")
print(df_merged[["area_km2", "pop_density", "log_pop_density"]].describe().T)