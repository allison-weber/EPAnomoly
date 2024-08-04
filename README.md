# DVA_project
HDDA Survivors group project for DVA Spring 2024

# Conda Environment Setup

We'll use a standardized conda environment so we can all run each others' code. To set this up, do the following in the root of this repository:

```
conda create -n dva_project
conda activate dva_project
python3 -m pip install -r requirements.txt
```

# Pulling the Data

In order for the application to work, the data must be pulled. For this, we have 2 scripts in the `data/` directory. All told, these scripts will create about 2.6 gigabytes of data on your machine.

```bash
conda activate dva_project # Activate conda environment if not currently active

cd data

python3 scrape_epa.py # Can take anywhere from 30-60 minutes to run.
```

This will scrape all of the raw data from the EPA into the following directory structure `{hourly,daily}/{sensor_type}/{year}.parquet`.

Since the EPA files are very large (75 GB+ for a single year in CSV format), we use the parquet format which achieves a > 100x reduction in file size. The drawback to this is that the data cannot be opened in a spreadsheet software like Excel.

Next, we need to partition the data by each site. This script has 2 steps:

1. Combine all of the yearly data into one combined file for each variable

2. Parititon these combined data into the directory structure `data/sites/{site_id}/{variable}.parquet`.

To run this script, first uncomment the block of code in the `Combining each site's dataframes into a single file` section:

```python
for freq in ["daily", "hourly"]:
    print(f"Combining {freq} dataframes")
    t0 = time.time()
    cols_to_keep = DAILY_COLS_TO_KEEP if freq == "daily" else HOURLY_COLS_TO_KEEP
    for var in VAR_TYPES:
        year_files = glob(f"{freq}/{var}/[0-9][0-9][0-9][0-9].parquet") # Get all of the files for the given year for a given variable
        if len(year_files) > 0:
            dfs = [pl.read_parquet(file, columns=cols_to_keep) for file in year_files]
            combined = pl.concat(dfs)
            combined = combined.with_columns((pl.col("State Code").cast(pl.Utf8) + pl.col("County Code").cast(pl.Utf8) + pl.col("Site Num").cast(pl.Utf8)).alias("site_id"))
            combined.write_parquet(f"{freq}/{var}/combined.parquet")
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
```

And then run the script

```bash
python3 partition_by_site.py
```

Give this another 10-20 minutes to run and the data pulling process will be complete! All said, this will take about **2.6 gigabytes of space.**

Next, run the script to pre-compute spline MSEs on the hourly data. This takes about 10 minutes when multiprocessed.

After that, run the script to pre-compute splines on daily data. This takes about 20 minutes (not multi-processed).

```bash
python3 fit_splines_hourly.py

python3 fit_splines_daily.py
```

# Running the Application

> Note: These are temporary instructions, just using this for basic development

Go to the `viz` directory and run `python3 epanomaly.py` while in the appropriate conda environment

Remember to install the `dash`, `plotly`, and `polars` packages, as they may not have been specified the last time you created the environment. The simplest way to do this is to run the following in the root of the repository:

```bash
python3 -m pip install -r requirements.txt
```

# Important Links

* EPA data download link: https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw

* Air Data Download documentation: https://aqs.epa.gov/aqsweb/airdata/FileFormats.html#_introduction