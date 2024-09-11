"""
2024-09-11 VMP

Feature engineering for entries. 
- Date range
- Entity tags
- Region data 

"""

import pandas as pd
import numpy as np

# load entry data
entry_data = pd.read_csv("../data/raw/entry_data.csv")

# again only work with groups
entry_data = entry_data[entry_data["poll_name"].str.contains("Group")]

# take data on date-range and region ID to merge with other data
entry_daterange = entry_data[["entry_id", "year_from", "year_to"]].drop_duplicates()

### region data ###

# load region data
region_data = pd.read_csv("../data/raw/region_data.csv")
region_data = region_data[["region_id", "world_region"]].drop_duplicates()
region_ids = entry_data[["entry_id", "region_id"]].drop_duplicates()

# merge
region_data = region_ids.merge(region_data, on="region_id", how="inner")

# pivot world region
entry_wr = region_data[["entry_id", "world_region"]].drop_duplicates()
entry_wr = entry_wr.pivot_table(
    index="entry_id", columns="world_region", aggfunc=len, fill_value=0
).reset_index()
entry_wr = entry_wr.rename_axis(None, axis=1)

# fix column names here
entry_wr = entry_wr.rename(
    columns={
        "Africa": "region_africa",
        "Central Eurasia": "region_central_eurasia",
        "East Asia": "region_east_asia",
        "Europe": "region_europe",
        "North America": "region_north_america",
        "Oceania-Australia": "region_oceania_australia",
        "South America": "region_south_america",
        "South Asia": "region_south_asia",
        "Southeast Asia": "region_southeast_asia",
        "Southwest Asia": "region_southwest_asia",
    }
)

# merge back region id
region_data = region_data[["entry_id", "region_id"]].drop_duplicates()
region_data = entry_wr.merge(region_data, on="entry_id", how="inner")

### entry tags ###

# load entity tags
entity_tags = pd.read_csv("../data/raw/entity_tags.csv")

# only take the ones that are in groups
entry_ids = entry_data["entry_id"].unique()
entity_tags = entity_tags[entity_tags["entry_id"].isin(entry_ids)]

# find the most used entity tags
n_entries = entry_data["entry_id"].nunique()
entity_tags = entity_tags[entity_tags["entrytag_level"] > 1]
entity_count = (
    entity_tags.groupby(["entrytag_id"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
entity_count = entity_count.head(10)

"""
18: Christian Traditions
24: Islamic Traditions
14: Buddhist Traditions
510: Indic Religious Traditions 
123: Chinese Religion
9: African Religions
38: Native American Religions
905: Abrahamic
508: South Americanm Religions
14828: Arabic
"""

# take top 10 (around 8% coverage for least used)
entity_tags = entity_tags[
    entity_tags["entrytag_id"].isin(entity_count["entrytag_id"].tolist())
]
entity_tags = entity_tags[["entry_id", "entrytag_id"]].drop_duplicates()

# pivot
entity_pivot = entity_tags.pivot_table(
    index="entry_id", columns="entrytag_id", aggfunc=len, fill_value=0
).reset_index()

# fix column names and index
entity_pivot = entity_pivot.rename_axis(None, axis=1)
entity_pivot.columns = ["entry_id"] + [
    "entrytag_{}".format(col) for col in entity_pivot.columns if col != "entry_id"
]

### merge entity tags, region data and date data ###

# merge left because we have missing cases in region data
entry_metadata = region_data.merge(entity_pivot, on="entry_id", how="left").fillna(0)

# fix floats to integers
for col in test.columns:
    if col != "entry_id":
        test[col] = test[col].astype(int)

# merge with date data
entry_metadata = entry_daterange.merge(entry_metadata, on="entry_id", how="inner")
entry_metadata = entry_metadata.sort_values(by="entry_id")
entry_metadata.to_csv("../data/preprocessed/entry_metadata.csv", index=False)
