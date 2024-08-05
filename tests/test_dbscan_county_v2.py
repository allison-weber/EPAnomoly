import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time


def load_daily_sensor(site_id:int, var:str) -> pl.DataFrame:
    df = pl.read_parquet(f"../src/data/daily/sensors/{site_id}/{var}.parquet")
    return df


def load_county_aqi_data_pl() -> pl.DataFrame:
    df = pl.read_csv(
        "../src/data/Combined_AQI_By_County.csv",
        try_parse_dates=True,null_values={'State Code': 'CC'})
    # return to pandas so rest of code functions without modification for now
    df = df.to_pandas()
    df = df[['Defining Site', 'Date', 'AQI']]
    return df


def plot_outliers(data: pd.DataFrame, core_sample_indices: np.array, site_id: str):
    # based on: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    labels = data['Cluster']
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True

    colors = [plt.cm.Spectral(each) for each in
              np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):

        if k == -1:
            col = [0, 0, 0, 1]  # Black used for noise.

        class_member_mask = labels == k

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy['Date'], xy['AQI'], "o", markerfacecolor=tuple(col),
                 markeredgecolor="k", markersize=8)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy['Date'], xy['AQI'], "o", markerfacecolor=tuple(col),
                 markeredgecolor="k", markersize=18)

    plt.title(f"DBSCAN: AQI for County {site_id}, "
              f"estimated number of clusters: {len(unique_labels) - 1}")
    plt.show()


def plot_outliers2(DBSCAN_dataset, site_id: str):
    # based on https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html
    fig2, (axes) = plt.subplots(1, 1, figsize=(12, 5))

    clusters = DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1]
    sns.scatterplot(data = clusters, x='Date', y='AQI', ax=axes,
                    hue='Cluster', palette='Set2', legend='full', s=10)
    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]
    axes.scatter(outliers['Date'], outliers['AQI'], s=200,
                 label='outliers', c="k")
    plt.title(f"DBSCAN: AQI for County {site_id}")
    plt.show()


def calcDBSCAN(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    # based on: https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html
    x_train = df.loc[:, ['AQI']]
    # scale data to increase speed
    scaler = MinMaxScaler()
    model = scaler.fit(x_train)
    x_train = model.transform(x_train)

    print(f"Start DBSCAN for county {site_id}...")
    # for now hard coding 2 key parameters
    clustering = DBSCAN(eps=.1, min_samples=3).fit(x_train)
    DBSCAN_dataset = df.copy()
    DBSCAN_dataset.loc[:, 'Cluster'] = clustering.labels_
    print(DBSCAN_dataset.Cluster.value_counts().to_frame())
    print(f"Outliers: {DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]}")

    # plot_outliers(DBSCAN_dataset, clustering.core_sample_indices_, site_id)
    return DBSCAN_dataset


def main(site_list: list = None):
    # sites = pl.read_parquet(f"../src/data/aqs_sites.parquet")
    # print(sites)

    # Polars is much faster than pandas to load csv
    # start = time.time()
    # df = load_county_aqi_data()
    # stop = time.time()
    # print(f"time to load data using pandas: {stop-start}")
    try:
        start = time.time()
        df = load_county_aqi_data_pl()
        stop = time.time()
        print(f"time to load data using polars: {stop - start}")
        if not site_list:
            site_list = (df['Defining Site']).unique()
        for site_id in site_list:
            county = df[df['Defining Site'] == site_id].dropna().reset_index(drop=True)
            DBSCAN_dataset = calcDBSCAN(county, site_id)
            plot_outliers2(DBSCAN_dataset, site_id)
    except FileNotFoundError:
        print("ERROR: County level csv file: data/Combined_AQI_By_County.csv not found."
              "\nAborting test. \nFind file on GitHub in Combined_AQI_By_County.part01.rar "
              "in data folder of rkhan87-patch-1 branch.")


if __name__ == '__main__':
    # ISSUES: when I originally did this for the progress report (see allison branch):
    # 1) I used 'County Code'- but that is NOT unique, so the results were all
    #   a compilation of counties. Semi-fixed here by using 'Defining Site'
    # 2) This used data from 1980-1998 (compiled files downloaded from EPA),
    #   however for the viz, we used data from 2018-2024 (+ did own AQI calc)
    sites = ["01-001-0002", "04-003-2003", "08-001-3001", "09-003-1123",
             "18-003-0012", "33-001-2003", "33-003-1002", "12-001-3011",
             "16-001-0016", "01-073-0024", "01-073-0012", "02-185-0041",
             "02-185-0042", "02-261-0004", "04-013-0013"]
    main(sites)
    # helpful DBSCAN overview articles:
    # https://medium.com/northraine/anomaly-detection-with-multi-dimensional-time-series-data-4fe8d111dee
    # https://www.turing.com/kb/time-series-anomaly-detection-in-python
    # DBSCAN docs:
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbsca