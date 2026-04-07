import geopandas as gpd
import pandas as pd
import numpy as np

from libpysal.weights import Queen
from esda.moran import Moran


# ----------------------------
# CONFIG: set your file paths
# ----------------------------
GEOJSON_PATH = "ri_tract.geojson"
RESIDUALS_CSV = "model_residuals.csv"   # should be in your project folder

# If your GeoJSON uses a different GEOID column name, set it here.
# Common alternatives: "GEOID20", "GEOID10", "geoid"
GEOID_COL_CANDIDATES = ["GEOID", "GEOID20", "GEOID10", "geoid"]


def pick_geoid_column(columns):
    for c in GEOID_COL_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(
        f"Could not find a GEOID column. Columns found: {list(columns)}. "
        f"Expected one of: {GEOID_COL_CANDIDATES}"
    )


def zfill_geoid(series: pd.Series) -> pd.Series:
    """
    Census tract GEOIDs are typically 11 digits (STATE2+COUNTY3+TRACT6).
    This pads with leading zeros if needed.
    """
    s = series.astype(str).str.strip()
    # If values sometimes have trailing '.0' due to numeric read, remove it:
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(11)


def main():
    # ----------------------------
    # Load tract geometry
    # ----------------------------
    gdf = gpd.read_file(GEOJSON_PATH)
    geoid_col = pick_geoid_column(gdf.columns)

    # Ensure we have geometry
    if "geometry" not in gdf.columns:
        raise ValueError("Geo file has no geometry column.")

    # Standardize GEOID name to "GEOID"
    if geoid_col != "GEOID":
        gdf = gdf.rename(columns={geoid_col: "GEOID"})

    gdf["GEOID"] = zfill_geoid(gdf["GEOID"])

    # ----------------------------
    # Load residuals
    # ----------------------------
    res = pd.read_csv(RESIDUALS_CSV)

    # Basic validation
    if "GEOID" not in res.columns or "residuals" not in res.columns:
        raise ValueError(
            f"{RESIDUALS_CSV} must contain columns: GEOID, residuals. "
            f"Found: {list(res.columns)}"
        )

    res["GEOID"] = zfill_geoid(res["GEOID"])
    res["residuals"] = pd.to_numeric(res["residuals"], errors="coerce")

    # Drop missing residual values
    res = res.dropna(subset=["residuals"])

    # ----------------------------
    # Merge (keep only matched tracts)
    # ----------------------------
    before_geo = len(gdf)
    before_res = len(res)

    gdfm = gdf.merge(res[["GEOID", "residuals"]], on="GEOID", how="inner")

    print("---- Merge diagnostics ----")
    print("Tracts in geo file:", before_geo)
    print("Rows in residuals:", before_res)
    print("Matched tracts used:", len(gdfm))

    if len(gdfm) < 30:
        print("\nWARNING: Very few matches. Likely GEOID formatting mismatch.")
        print("Example GEOIDs in geo file:", gdf["GEOID"].head().tolist())
        print("Example GEOIDs in residuals:", res["GEOID"].head().tolist())
        return

    # Remove invalid / empty geometries (rare, but can break weights)
    gdfm = gdfm[~gdfm.geometry.is_empty & gdfm.geometry.notna()].copy()

    # (Optional) ensure projected CRS not required for contiguity; works in any CRS.
    # But if you later do distance-based weights, you must project.

    # ----------------------------
    # Build spatial weights (Queen contiguity)
    # ----------------------------
    w = Queen.from_dataframe(gdfm, silence_warnings=True)

    # Identify islands (tracts with no neighbors)
    islands = list(w.islands)
    print("\n---- Weights diagnostics ----")
    print("N tracts in weights:", w.n)
    print("Islands (no neighbors):", len(islands))

    # If there are islands, drop them for Moran's I (recommended)
    if len(islands) > 0:
        print("Dropping islands for Moran's I...")
        gdfm = gdfm.drop(index=islands).copy()
        w = Queen.from_dataframe(gdfm, silence_warnings=True)

    # Row-standardize weights
    w.transform = "r"

    # ----------------------------
    # Moran's I on residuals
    # ----------------------------
    x = gdfm["residuals"].values
    # Safety: no NaNs
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Residuals contain NaN/Inf after merge. Check your residuals file.")

    mi = Moran(x, w, permutations=999)

    print("\n---- Moran's I results (residuals) ----")
    print("Moran's I:", mi.I)
    print("Expected I:", mi.EI)
    print("z-score:", mi.z_sim)
    print("p-value (permutation):", mi.p_sim)

    if mi.p_sim < 0.05:
        print("\nINTERPRETATION: Significant spatial autocorrelation in residuals (p < 0.05).")
        print("Next step: consider a spatial model (CAR/BYM, spatial error, or add spatial random effects).")
    else:
        print("\nINTERPRETATION: No significant spatial autocorrelation in residuals (p >= 0.05).")
        print("Your Poisson GLM residuals look spatially independent enough for standard inference.")

    # Optional: save the merged file for mapping/debugging
    gdfm.to_file("tracts_with_residuals.geojson", driver="GeoJSON")
    print("\nSaved merged file: tracts_with_residuals.geojson")


if __name__ == "__main__":
    main()
