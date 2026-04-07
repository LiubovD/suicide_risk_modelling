import pandas as pd

# -----------------------------
# FILE PATHS (change if needed)
# -----------------------------
SUICIDE_FILE = "suicide_summary_by_GEOID.csv"
ACS_FILE = "outputs/ri_tract_acs_features_2023.csv"
OUTPUT_FILE = "outputs/suicides_by_tract_2019_2023_with_acs.csv"


# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")

suicides = pd.read_csv(SUICIDE_FILE, dtype={"GEOID": str}, low_memory=False)
acs = pd.read_csv(ACS_FILE, dtype={"GEOID": str}, low_memory=False)

# ensure GEOID format matches
suicides["GEOID"] = suicides["GEOID"].str.replace(r"\.0$", "", regex=True).str.zfill(11)
acs["GEOID"] = acs["GEOID"].str.replace(r"\.0$", "", regex=True).str.zfill(11)

print("Suicide rows:", len(suicides))
print("ACS rows:", len(acs))


# -----------------------------
# JOIN BY GEOID
# -----------------------------
print("Merging by GEOID...")

merged = suicides.merge(
    acs,
    on="GEOID",
    how="left"   # keeps ALL suicide rows
)


# -----------------------------
# SAVE RESULT
# -----------------------------
merged.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print("Saved:", OUTPUT_FILE)
print("Columns added:", len(acs.columns) - 1)