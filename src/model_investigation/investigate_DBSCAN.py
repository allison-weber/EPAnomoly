import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
from cusum_test import load_county_aqi_data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time


def load_daily_sensor(site_id:int, var:str) -> pl.DataFrame:
    df = pl.read_parquet(f"../data/daily/sensors/{site_id}/{var}.parquet")
    return df


def load_county_aqi_data_pl() -> pl.DataFrame:
    df = pl.read_csv(
        "../data/Combined_AQI_By_County.part01/Combined_AQI_By_County.csv",
        try_parse_dates=True,null_values={'State Code': 'CC'})
    # return to pandas so rest of code functions without modification for now
    df = df.to_pandas()
    df = df[['County Code', 'Date', 'AQI']]
    df["County Code"] = df["County Code"].astype('Int64')
    return df


def plot_outliers(data: pd.DataFrame, core_sample_indices: np.array, county_num: int):
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

    plt.title(f"DBSCAN: AQI for County {county_num}, "
              f"estimated number of clusters: {len(unique_labels) - 1}")
    plt.show()


def plot_outliers2(DBSCAN_dataset, county_num: int = 1):
    # based on https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html
    fig2, (axes) = plt.subplots(1, 1, figsize=(12, 5))

    clusters = DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1]
    sns.scatterplot(data = clusters, x='Date', y='AQI', ax=axes,
                    hue='Cluster', palette='Set2', legend='full', s=10)
    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]
    axes.scatter(outliers['Date'], outliers['AQI'], s=200,
                 label='outliers', c="k")
    plt.title(f"DBSCAN: AQI for County {county_num}")
    plt.show()


def calcDBSCAN(df: pd.DataFrame, county_num: int = 1) -> pd.DataFrame:
    # based on: https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html
    x_train = df.loc[:, ['AQI']]
    # scale data to increase speed
    scaler = MinMaxScaler()
    model = scaler.fit(x_train)
    x_train = model.transform(x_train)

    print(f"Start DBSCAN for county {county_num}...")
    # for now hard coding 2 key parameters
    clustering = DBSCAN(eps=.01, min_samples=3).fit(x_train)
    DBSCAN_dataset = df.copy()
    DBSCAN_dataset.loc[:, 'Cluster'] = clustering.labels_
    print(DBSCAN_dataset.Cluster.value_counts().to_frame())
    print(f"Outliers: {DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]}")

    # plot_outliers(DBSCAN_dataset, clustering.core_sample_indices_, county_num)
    return DBSCAN_dataset


def main(county_num: int = 1):
    sites = pl.read_parquet(f"../data/aqs_sites.parquet")
    # print(sites)

    # Polars is much faster than pandas to load csv
    # start = time.time()
    # df = load_county_aqi_data()
    # stop = time.time()
    # print(f"time to load data using pandas: {stop-start}")

    start = time.time()
    df = load_county_aqi_data_pl()
    stop = time.time()
    print(f"time to load data using polars: {stop-start}")

    county = df[df['County Code'] == county_num].dropna().reset_index(drop=True)
    DBSCAN_dataset = calcDBSCAN(county, county_num)
    plot_outliers2(DBSCAN_dataset, county_num)


if __name__ == '__main__':
    COUNTY = 1  # For demo purposes just running 1 county at a time
    main(COUNTY)
    # helpful DBSCAN overview articles:
    # https://medium.com/northraine/anomaly-detection-with-multi-dimensional-time-series-data-4fe8d111dee
    # https://www.turing.com/kb/time-series-anomaly-detection-in-python
    # DBSCAN docs:
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
