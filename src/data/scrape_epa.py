# -*- coding: utf-8 -*-
'''
Scrapes the EPA website to download all of the hourly and daily data
for all of their air pollutants.

Target date range: 2010-present.

They technically have an API, but honestly, it seems easier to just scrape

Saves daily data into the daily/ directory and hourly data into the hourly/ directory.
'''
import backoff
import os
import pandas as pd
import requests
import zipfile

# Define the data types
# Files are specified using the following format:
# {hourly|daily}_{key}_{year}.csv
# where key corresponds to the keys of the following dict:
DATA_TYPES = {
    # Gases
    "44201": "Ozone",
    "42401": "SO2",
    "42101": "CO",
    "42602": "NO2",
    # Particulates
    "88101": "PM2.5 FRM",
    "88502": "PM2.5 non-FRM",
    "81102": "PM10",
    "86101": "PMc",
    # Meteorological
    "WIND": "Wind",
    "TEMP": "Temperature",
    "PRESS": "Pressure",
    "RH_DP": "RH_DP",
    # Toxics
    "HAPS": "HAPs",
    "VOCS": "VOCs",
    "NONOxNOy": "NONOxNOy"
}

DAILY_SCHEMA = {
    "State Code"            : "UInt32",
    "County Code"           : "UInt32",
    "Site Num"              : "UInt32",
    "Parameter Code"        : "UInt32",
    "POC"                   : "UInt32",
    "Latitude"              : "Float64",
    "Longitude"             : "Float64",
    "Datum"                 : "string",
    "Parameter Name"        : "string",
    "Sample Duration"       : "string",
    "Pollutant Standard"    : "string",
    "Date Local"            : "string",
    "Units of Measure"      : "string",
    "Event Type"            : "string",
    "Observation Count"     : "UInt32",
    "Observation Percent"   : "Float64",
    "Arithmetic Mean"       : "Float64",
    "1st Max Value"         : "Float64",
    "1st Max Hour"          : "UInt32",
    "AQI"                   : "UInt32",
    "Method Code"           : "UInt64",
    "Method Name"           : "string",
    "Local Site Name"       : "string",
    "Address"               : "string",
    "State Name"            : "string",
    "County Name"           : "string",
    "City Name"             : "string",
    "CBSA Name"             : "string",
    "Date of Last Change"   : "string",
}

HOURLY_SCHEMA = {
    "State Code"            : "UInt32",
    "County Code"           : "UInt32",
    "Site Num"              : "UInt32",
    "Parameter Code"        : "UInt32",
    "POC"                   : "UInt32",
    "Latitude"              : "Float64",
    "Longitude"             : "Float64",
    "Datum"                 : "string",
    "Parameter Name"        : "string",
    "Date Local"            : "string",
    "Time Local"            : "string",
    "Date GMT"              : "string",
    "Time GMT"              : "string",
    "Sample Measurement"    : "Float64",
    "Units of Measure"      : "string",
    "MDL"                   : "Float64",
    "Uncertainty"           : "Float64",
    "Qualifier"             : "string",
    "Method Type"           : "string",
    "Method Code"           : "UInt32",
    "Method Name"           : "string",
    "State Name"            : "string",
    "County Name"           : "string",
    "Date of Last Change"   : "string"
}

SITE_DESCRIPTION_SCHEMA = {
    "State Code"            : "UInt32",
    "County Code"           : "UInt32",
    "Site Number"           : "UInt32",
    "Latitude"              : "Float64",
    "Longitude"             : "Float64",
    "Datum"                 : "string",
    "Elevation"             : "Float32",
    "Land Use"              : "string",
    "Location Setting"      : "string",
    "Site Established Date" : "string",
    "Site Closed Date"      : "string",
    "Met Site State Code"   : "Float64",
    "Met Site County Code"  : "Float64",
    "Met Site Site Number"  : "Float64",
    "Met Site Type"         : "string",
    "Met Site Distance"     : "Float64",
    "Met Site Direction"    : "string",
    "GMT Offset"            : "Int32",
    "Owning Agency"         : "string",
    "Local Site Name"       : "string",
    "Address"               : "string",
    "Zip Code"              : "UInt32",
    "State Name"            : "string",
    "County Name"           : "string",
    "City Name"             : "string",
    "CBSA Name"             : "string",
    "Tribe Name"            : "string",
    "Extraction Date"       : "string"
}

def _convert_csv_to_parquet(fname: str, freq: str = "daily", schema: dict = None) -> None:
    '''
    Converts the given CSV file to a parquet file.
    '''
    if schema is None:
        schema = DAILY_SCHEMA if freq == "daily" else HOURLY_SCHEMA
    
    df = pd.read_csv(fname, dtype=schema)
    df.to_parquet(fname.replace(".csv", ".parquet"))
    df_pq = pd.read_parquet(fname.replace(".csv", ".parquet"))
    if df.equals(df_pq):
        os.remove(fname)

def _fatal_code(e):
    # We don't want to retry on 400 errors
    return 400 <= e.response.status_code < 500

# These translate into the following URLs:
# https://aqs.epa.gov/aqsweb/airdata/{hourly|daily}_{key}_{year}.zip

@backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=5,
        giveup=_fatal_code)
def _send_request(url: str) -> requests.Response:
    '''
    Sends a request to the given URL. Wrapper in order to use backoff.
    '''
    return requests.get(url)


def get_zip_file(data_type: str = "44201", year: str = "2023", freq: str = "daily") -> None:
    '''
    Downloads the zip file for the given data type and year.
    '''
    url = f"https://aqs.epa.gov/aqsweb/airdata/{freq}_{data_type}_{year}.zip"

    # Download the file
    r = _send_request(url)

    # Write the file to a zip archive
    with open(f"{freq}_{data_type}_{year}.zip", "wb") as f:
        f.write(r.content)

    # check if the {freq}/{data_type} directory exists
    if not os.path.exists(f"{freq}/{DATA_TYPES[data_type]}"):
        os.makedirs(f"{freq}/{DATA_TYPES[data_type]}")
    
    # Unzip the file
    with zipfile.ZipFile(f"{freq}_{data_type}_{year}.zip", "r") as zip_ref:
        zip_ref.extractall(f"{freq}/{DATA_TYPES[data_type]}")

    # Convert the CSV to parquet
    for fname in zip_ref.namelist():
        if fname.endswith(".csv"):
            _convert_csv_to_parquet(f"{freq}/{DATA_TYPES[data_type]}/{fname}", freq=freq)
    
    os.remove(f"{freq}_{data_type}_{year}.zip")


def main():
    if not os.path.exists("daily"):
        os.makedirs("daily")
    
    if not os.path.exists("hourly"):
        os.makedirs("hourly")

    # Download the site information file and convert it to parquet
    # Creates the file data/aqs_sites.parquet
    url = "https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip"
    r = requests.get(url)
    r.raise_for_status()

    with open("site_list.zip", "wb") as f:
        f.write(r.content)
    
    with zipfile.ZipFile("site_list.zip", "r") as zip_ref:
        zip_ref.extractall("site_list")

    df = pd.read_csv("site_list/aqs_sites.csv")

    # Get rid of sites not in the mainland US:
    # canadian ("CC");  Mexican ("80"); Virign Islands ("78");
    # Puerto Rico ("72"); Guam ("66")
    df = df[~df["State Code"].isin(["CC", "80", "78", "72", "66"])]
    df = df.astype(SITE_DESCRIPTION_SCHEMA)
    df.to_parquet("aqs_sites.parquet")

    os.remove("site_list.zip")
    os.remove("site_list/aqs_sites.csv")
    os.rmdir("site_list")

    # Download the daily and hourly files for each data type
    for key in DATA_TYPES:
        for year in range(2018, 2024):
            get_zip_file(data_type=key, year=str(year), freq="daily")
            get_zip_file(data_type=key, year=str(year), freq="hourly")

    # Rename the files to just the year since the rest of the information is in the directory structure
    for freq in ["daily", "hourly"]:
        for key in DATA_TYPES:
            for year in range(2018, 2024):
                os.rename(f"{freq}/{DATA_TYPES[key]}/{freq}_{key}_{year}.parquet", f"{freq}/{DATA_TYPES[key]}/{year}.parquet")
                

if __name__ == '__main__':
    main()