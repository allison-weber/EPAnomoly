import polars as pl
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import sort_graph_by_row_values
from multiprocessing import Pool, cpu_count


MIN_POINTS = 4


def add_color_column(data: pl.DataFrame) -> pl.DataFrame:
    """
    Add variable to make coloring outlier points on map easy
    :param data: Data frame with outlier column
    :return: copy of dataframe with column to help color map
    """
    df = data.with_columns(pl.when(pl.col("outlier") != -1)
                           .then(pl.lit('No'))
                           .otherwise(pl.lit('Yes'))
                           .alias("DBSCAN anomaly detected?"))
    return df


def create_clusters_dbscan(data: pl.DataFrame, col_name: str):
    """
    Performs DBSCAN, scaling to optimize for speed
    Note: params + function will need to be updated to use more than 1 col
    :param data: data for a single site
    :param col_name: variable used to look for outliers
    :return:
    """
    # when creating line plots, check for min # of points
    if data.shape[0] <= MIN_POINTS:
        return np.zeros(data.shape[0])
    # Min Max scaling using polars:
    # https://stackoverflow.com/questions/74398563/how-to-use-polars-dataframes-with-scikit-learn
    transformer = ColumnTransformer(
        transformers=[("scaled_var", MinMaxScaler(), [col_name])])
    df = transformer.fit_transform(data)
    transformer.set_output(transform="polars")
    df = np.round(df, 2)    # failed attempt to speed up DBSCAN, increase decimals to match eps
    # try to speed up by precompute nearest neighbors - not implement right
    # dist = 0.05
    # neigh = NearestNeighbors(radius=dist)
    # neigh.fit(df)
    # df_neigh = neigh.radius_neighbors_graph(df, mode='distance')
    # df_neigh = sort_graph_by_row_values(df_neigh)
    #clustering = DBSCAN(eps=dist, min_samples=3, metric='precomputed').fit(df_neigh)

    # TODO optimize params: eps and min_samples; adjust on site, variable, date range
    clustering = DBSCAN(eps=0.1, min_samples=3).fit(df) #  n_jobs=-1 doesn't seem to help, probably polars already using all avaible
    return clustering


def find_site_outliers_dbscan(data: pl.DataFrame) -> pl.dataframe:
    """
    DBSCAN anomaly detection for 1 site if at least 4 data points.
        0=No outlier, 1=Yes outlier
    :param data: data for a single site
    :return: 1 row dataframe with indicator column for if anomaly detected
    """
    # set initial result to no-outlier found (0)
    site_id = data.select(pl.first("site_id")).item()
    col_name = data.select(pl.first("col_name")).item()
    result = {"site_id": site_id, "outlier": 0, "DBSCAN anomaly detected?": 'No'}
    if data.shape[0] > MIN_POINTS:
        clustering = create_clusters_dbscan(data, col_name)
        if -1 in clustering.labels_:
            result = {"site_id": site_id, "outlier": -1, "DBSCAN anomaly detected?": 'Yes'}
    return result


def detect_anomaly_dbscan(data: pl.DataFrame, variable: str) -> pl.DataFrame:
    """
    Basic implementation of DBSCAN, 0=No anomaly, 1=anomaly detected
    :param data: dataframe with data for all sites
    :param variable: air quality measure evaluating for anomalies
    :return: dataframe with site ids and anomaly detection indicator column
    """
    start = time.time()
    col_name = "Arithmetic Mean" if variable != 'AQI' else 'AQI'

    print("Starting DBSCAN...")
    df = data.select(["site_id", col_name])
    df = df.with_columns(pl.lit(col_name).alias("col_name"))
    df_list = df.partition_by("site_id")
    chunk_size = 5
    num_processes = int(min(cpu_count() - 2, len(df_list) / chunk_size)) # avoid using all cores
    num_processes = cpu_count() if num_processes < 1 else num_processes
    with Pool(processes=num_processes) as p:
        result = p.map(find_site_outliers_dbscan, df_list, chunksize=chunk_size)  # alt: map vs imap; with/out chunksize
    df = pl.DataFrame(result)
    stop = time.time()
    print(f"total time to run dbscan for {variable} {num_processes=} {chunk_size=}: {stop - start}")
    return df


def main():
    # For testing, none of this code is loaded in when the script is imported
    print("Testing moved to test_dbscan.py.")


if __name__ == "__main__":
    main()
