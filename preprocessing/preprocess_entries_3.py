"""
VMP 2023-01-19:
Decisions: 
1. (year_from, year_to): earliest observed to latest observed. 
2. (region_id): region used for most questions per entry
"""

import pandas as pd
import numpy as np

# basic imports
df = pd.read_csv("../data/raw/raw_data.csv")
gis_regions = pd.read_csv("../data/raw/gis_regions.csv")

# only take polls that are relevant
df = df[df["poll"].isin(["Religious Group (v6)", "Religious Group (v5)"])]

# select columns related to entry metadata
entry_metadata_cols = ["entry_id", "entry_name", "year_from", "year_to", "region_id"]
df_entries = df[entry_metadata_cols].drop_duplicates()

### mode region ###
from functions_entries import mode_feature_by_entry

mode_region = mode_feature_by_entry(df, "entry_id", "region_id")
mode_region = mode_region[["entry_id", "region_id"]]

# get the area associated with gis regions
from functions_kml import split_regions, calculate_gis_area

# calculate gis
assert len(gis_regions) == gis_regions["region_id"].nunique()
gis_metadata = split_regions(gis_regions)
region_area = calculate_gis_area(gis_metadata)

# save gis metadata for regions coded
gis_metadata.to_csv("../data/preprocessed/gis_metadata.csv", index=False)

# extract and clean area
region_area = region_area[["region_id", "region_area"]].drop_duplicates()
region_area = region_area.groupby("region_id")["region_area"].sum()
region_area = region_area.reset_index(name="region_area")
region_area["log_area"] = np.log(region_area["region_area"])
region_area = region_area.drop(columns="region_area")

# merge with entries
mode_region = mode_region.merge(region_area, on="region_id", how="left")

### timespan ###
mode_timespan = mode_feature_by_entry(df, "entry_id", ["year_from", "year_to"])
mode_timespan = mode_timespan[["entry_id", "year_from", "year_to"]]
assert len(mode_timespan) == df_entries["entry_id"].nunique()
assert mode_timespan["entry_id"].nunique() == df_entries["entry_id"].nunique()

### merge and add back entry_name ###
entry_name_id = df_entries[["entry_id", "entry_name"]].drop_duplicates()
df_entries_clean = mode_timespan.merge(mode_region, on="entry_id", how="inner")
df_entries_clean = entry_name_id.merge(df_entries_clean, on="entry_id", how="inner")
df_entries_clean["year_from"] = df_entries_clean["year_from"].astype(int)
df_entries_clean["year_to"] = df_entries_clean["year_to"].astype(int)
df_entries_clean = df_entries_clean.sort_values(by="entry_id")
df_entries_clean.to_csv("../data/preprocessed/entry_metadata.csv", index=False)
