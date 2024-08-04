# Scrape the data from the EPA website and compress it (save to parquet, optimize schema, etc.)
python3 scrape_epa.py

# Partition the data by site and compute the air quality data
python3 partition_by_site.py

# Precompute the splines for the hourly data
python3 fit_splines_hourly.py

# Precompute the splines for the daily data
python3 fit_splines_daily.py