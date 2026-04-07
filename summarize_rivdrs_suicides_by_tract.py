import pandas as pd
import os


# Read input file
df = pd.read_csv("input_suicide.csv")

summary_table = (
    df[
        (df["suicidedeath"] == "1") &
        (df["IncidentYear"].isin([2019, 2020, 2021, 2022, 2023]))
    ]
    .groupby("GEOID")
    .size()
    .reset_index(name="suicide_count")
)

# Save output file
summary_table.to_csv("suicide_summary_by_GEOID.csv", index=False)
