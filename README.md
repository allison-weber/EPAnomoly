# EPAnomoly

EPAnomaly provides an interactive visualization tool and on-the-fly anomaly detection for 16 environmental factors across the United States,using data scraped from the United States Environmental Protection Agency (EPA). It is run locally through a Plotly Dash application, and is largely implemented in Python. It was created by the 'HDDA Survivors' group as a project the Georgia Tech Master's of Analytics course Data and Visual Analytics in Spring 2024.

## My Contribution
- Researched different outlier detection methodologies, including literature review, see `docs/DVA-models+algorithms.docx`.
- Created simple test implementations for 3 of the 4 methods the team choose for further evaluation, DBSCAN, CUSUM and ARIMA (investigation into final B-splines performed by another teammate), code in `src/model_investigation` folder and initial plots in `docs/DVA model development.pptx` 
- Implemented all code to detect outliers using DBSCAN including adjustements needed based on user-selection of air quality variable, date-range, site, etc. 
- Integrated results of DBSCAN into the Dash app, including marking outliers on main map and on individual site line plots  
- Developed test code to test DBSCAN implementation correctness and also time tests to evaluate for scalability and usability
- Supported the team with other miscellaneous tasks, performing code reviews, fixing bugs, writing sections of final report, etc. 

## Overview

The application itself is a dashboard with the following features:
1. View and select for further inspection any EPA environmental sensor across the continental United States, with outliers highlighted
2. Filter the data to specific date ranges
3. Select an anomaly detection model from 3 presets, which performs anomaly detection in real time on the subset of dates you selected.
4. Clicking on a site allows you select different air quality variables measured at that site to be shown in time series charts to see how they change over your selected date range, with outliers hightlighed.

Image below provides a quick overview of the EPAnomaly dashboard appearance: 
- At left is the main map with sites where outliers are detected (in the selected date range using the daily B-spline method) highlighted in red (blue for sites with no anomolies detected). 
- At right is an example of the line plot of the Air Quality Index (AQI) for 1 particular site with the outliers highlighted in red.   

![dashboard_snap_v2](https://github.com/user-attachments/assets/ae69c3ec-aad5-4e44-88b6-9cb86e32aa5b)



## Repo structure

The source code all lives inside the src/ directory. This directory is split into two sub-directories: data/ and viz/.
The data/ directory contains all of the setup code that is used to scrape and process the data, and the viz/ directory
contains the application itself, including all of the visualization and on-the-fly anomaly detection code.

Instructions are below for getting started.


## Conda environment setup

We recommend using a conda environment to run the code. To do so, navigate to the src directory

Now, within the src/ directory, run the following commands in your terminal:


```
conda create -n dva_project # You can name it anything you like, but just be consistent
conda activate dva_project
python3 -m pip install -r requirements.txt
```

## Pulling and preprocesssing the data

In order for the application to work, some scripts must be run to perform the following tasks:

1. Scrape the data from the EPA website

2. Partition the data by site

3. Precompute hourly spline mean squared errors for each day

4. Precompute daily spline interpolation errors for each day

All told, these scripts will create about 2.6 gigabytes of data on your machine. In order, they are:

```
conda activate dva_project # Activate conda environment if not currently active

cd data # if not already in the src/data directory

python3 scrape_epa.py # Can take 30-60 minutes to run.

python3 partition_by_site.py # Can take 10-20 minutes to run

python3 fit_splines_hourly.py # Can take 15-30 minutes to run

python3 fit_splines_daily.py # Can take 30-60 minutes to run
```

## Running the Application

Go to the `viz` directory and run `python3 epanomaly.py` while in the appropriate conda environment.

From there, the console output will give you a link. Paste that link to your browser, and you will be off to the races!

By default, this should be http://127.0.0.1:8050/

## Important Links

* EPA data download link: https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw

* Air Data Download documentation: https://aqs.epa.gov/aqsweb/airdata/FileFormats.html#_introduction

## HDDA Survivors Team Members
Allison Weber	(me)  
Marco Segura Camacho  
Raaed Mahmud Khan  
Calista Randazzo  
Abhinuv Uppal  
Ben Yu  

