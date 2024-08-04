#-*- coding: utf-8 -*-
'''
Combines the EPA data into a single file for each sensor type.

Then, for each site, combines all of the data for that site into a single directory.

This leads to 2800-ish directories, in the form of data/sensors/{hourly,daily}/{site_id}/{variable_type}.parquet. This makes
data easy to read for the visualization (particularly clicking into a site). 
'''

import os
import polars as pl
import time

from glob import glob

DAILY_COLS_TO_KEEP = [
    "Date Local",
    "State Code",
    "County Code",
    "Site Num",
    "Arithmetic Mean",
]

HOURLY_COLS_TO_KEEP = [
    "Date Local",
    "Time Local",
    "State Code",
    "County Code",
    "Site Num",
    "Sample Measurement",
]

VAR_TYPES = [
    "Ozone",
    "SO2",
    "CO",
    "NO2",
    "PM2.5 FRM",
    "PM2.5 non-FRM",
    "PM10",
    "PMc",
    "Wind",
    "Temperature",
    "Pressure",
    "RH_DP",
    "HAPs",
    "VOCs",
    "NONOxNOy"
]

SITES_COLS_TO_KEEP = [
    'State Code',
    'County Code',
    'Site Number',
    'Latitude',
    'Longitude',
    'Local Site Name',
    'Address',
    'Zip Code',
    'State Name',
    'County Name',
    'City Name',
    'CBSA Name'
]

AQI_VARS = [
    # Variables which also have AQI values in their data
    "Ozone",
    "SO2",
    "CO",
    "NO2",
    "PM2.5 FRM",
    "PM2.5 non-FRM",
    "PM10",
]

##################################################################
###   Combining each variable's dataframes into a single file  ###
##################################################################

# Uncomment if columns to keep have changed or if you have not run this code before
# Takes about 45 seconds on my machine (Abhi)

for freq in ["daily", "hourly"]:
    print(f"Combining {freq} dataframes")
    t0 = time.time()
    for var in VAR_TYPES:
        # Keep the AQI column when available
        cols_to_keep = DAILY_COLS_TO_KEEP.copy() if freq == "daily" else HOURLY_COLS_TO_KEEP.copy()
        if var in AQI_VARS and freq == "daily":
            cols_to_keep.append("AQI")
        
        year_files = glob(f"{freq}/{var}/[0-9][0-9][0-9][0-9].parquet") # Get all of the files for the given year for a given variable
        if len(year_files) > 0:
            dfs = [pl.read_parquet(file, columns=cols_to_keep) for file in year_files]
            combined = pl.concat(dfs)
            combined = combined.with_columns((pl.col("State Code").cast(pl.Utf8) + pl.col("County Code").cast(pl.Utf8) + pl.col("Site Num").cast(pl.Utf8)).alias("site_id"))
            # dedupe
            if freq == "daily":
                combined = combined.unique(subset=["Date Local", "site_id"], keep="first")
            else:
                combined = combined.unique(subset=["Date Local", "Time Local", "site_id"], keep="first")
            combined.write_parquet(f"{freq}/{var}/combined.parquet")
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")

#####################################
###   CLEANING SITE INFORMATION   ###
#####################################

if not os.path.exists("daily/sites"):
    os.makedirs("daily/sites")

if not os.path.exists("hourly/sites"):
    os.makedirs("hourly/sites")

print("===   Partitioning Daily Data   ===")

for var in VAR_TYPES:
    cols_to_keep = ["Date Local", "site_id", "Arithmetic Mean"]

    df = pl.read_parquet(f"daily/{var}/combined.parquet", columns=cols_to_keep)\
        .rename({"Arithmetic Mean": var})\
        .select(["Date Local", "site_id", var])\

    # Save the data for each site in df under sites/{site_id}/{var}.parquet
    site_ids = df["site_id"].unique().to_list()
    print(f"Number of sites for {var}: {len(site_ids)}")
    for site_id in site_ids:
        if not os.path.exists(f"daily/sites/{site_id}"):
            os.makedirs(f"daily/sites/{site_id}")
        
        df.filter(pl.col("site_id") == site_id)\
            .drop("site_id")\
            .sort("Date Local")\
            .write_parquet(f"daily/sites/{site_id}/{var}.parquet")
        
    print(f"Finished {var}")

print("===   Partitioning Hourly Data   ===")

for var in VAR_TYPES:
    cols_to_keep = ["Date Local", "Time Local", "site_id", "Sample Measurement"]

    df = pl.read_parquet(f"hourly/{var}/combined.parquet", columns=cols_to_keep)\
        .rename({"Sample Measurement": var})\
        .select(["Date Local", "Time Local", "site_id", var])\

    # Save the data for each site in df under sites/{site_id}/{var}.parquet
    site_ids = df["site_id"].unique().to_list()
    print(f"Number of sites for {var}: {len(site_ids)}")
    for site_id in site_ids:
        if not os.path.exists(f"hourly/sites/{site_id}"):
            os.makedirs(f"hourly/sites/{site_id}")
        
        df.filter(pl.col("site_id") == site_id)\
            .drop("site_id")\
            .sort("Date Local", "Time Local")\
            .write_parquet(f"hourly/sites/{site_id}/{var}.parquet")
        
    print(f"Finished {var}")

# Now we need to create a combined.parquet for the AQI data
print("===   Combining AQI Data   ===")

# Start with the Ozone data
aqi_df = pl.read_parquet("daily/Ozone/combined.parquet").select(["Date Local", "site_id", "AQI"])\
    .rename({"AQI": "AQI_Ozone"})\
    .unique(subset=["Date Local", "site_id"])

# Now join the rest of the AQI data
for var in AQI_VARS[1:]:
    join_df = pl.read_parquet(f"daily/{var}/combined.parquet")\
        .select(["Date Local", "site_id", "AQI"])\
        .rename({"AQI": f"AQI_{var}"})\
        .unique(subset=["Date Local", "site_id"])
    aqi_df = aqi_df.join(join_df, on=["Date Local", "site_id"], how="outer_coalesce")

# Now we need to find the max AQI value for each (site_id, Date Local) pair
AQI_COLNAMES = [f"AQI_{var}" for var in AQI_VARS]
aqi_df = aqi_df.with_columns(pl.max_horizontal(*AQI_COLNAMES).alias("AQI")).sort("site_id", "Date Local")\
    .select("Date Local", "site_id", "AQI")

# Create an AQI combined.parquet
if not os.path.exists("daily/AQI"):
    os.makedirs("daily/AQI")
aqi_df.write_parquet("daily/AQI/combined.parquet")

# Optional check to see if any (site_id, Date Local) pairs are duplicated
# gb = aqi_df.groupby(["site_id", "Date Local"]).agg(pl.count())
# print(gb.sort("count").filter(pl.col("count") > 1))

# Save the data for each site in df under sites/{site_id}/{var}.parquet
site_ids = aqi_df["site_id"].unique().to_list()
print(f"Number of sites for AQI: {len(site_ids)}")
for site_id in site_ids:
    if not os.path.exists(f"daily/sites/{site_id}"): # Probably already exists, but just in case
        os.makedirs(f"daily/sites/{site_id}")
    
    aqi_df.filter(pl.col("site_id") == site_id)\
        .drop("site_id")\
        .sort("Date Local")\
        .write_parquet(f"daily/sites/{site_id}/AQI.parquet")

print("Finished AQI")