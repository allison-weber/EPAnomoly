import sys
# need this to be able to read from the parent directory (import viz.*)
sys.path.append("..") 

import time
import polars as pl
import numpy as np
from multiprocessing import Pool
from viz.viz_models import detect_anomaly_dbscan, create_clusters_dbscan
from viz.viz_utils import _get_data_for_variable, _get_site_data

# Should really use pytest or other unit testing framework...
# but I'm going quick and dirty for now
NUM_TRIALS_PER_VARIABLE = 10

BENCHMARK_VARIABLES = [
    "AQI",
    "PM2.5 FRM",
    "PM2.5 non-FRM",
    "PM10",
    "CO"
]


def f(df):
    return df.shape


def pool_example(data_list):
    p = Pool(processes=3)
    result = p.map(f, data_list)
    p.close()
    p.join()
    print(result)


def create_simple_test_data():
    d1 = pl.DataFrame(
        {"AQI": [1, 2], "Arithmetic Mean": [3, 4], "site_id": [1, 1]})
    d2 = pl.DataFrame({"AQI": [1, 2, 3, 4], "Arithmetic Mean": [3, 4, 4, 6],
                       "site_id": [2, 2, 2, 2]})
    d3 = pl.DataFrame({"AQI": [1, 2, 3, 4, 5, 6, 100],
                       "Arithmetic Mean": [3, 4, 4, 6, 6, 6, 100],
                       "site_id": [3, 3, 3, 3, 3, 3, 3]})
    test_list = [d1, d2, d3]
    new_df = pl.concat(test_list, rechunk=True)
    return new_df


def load_county_aqi_data(site_list) -> pl.DataFrame:
    df = pl.read_csv(
        "../data/Combined_AQI_By_County.csv",
        try_parse_dates=True,
        null_values={'State Code': 'CC'})
    df = df.filter(pl.col("Defining Site").is_in(site_list))
    df = df.drop(["Category", "Defining Parameter", ''])
    return df


def test_dbscan_syntax():
    new_df = create_simple_test_data()
    df_list = new_df.partition_by("site_id")
    pool_example(df_list)  # use test_list or df_list
    detect_anomaly_dbscan(new_df, "AQI")


def test_model_dev(site_list):
    aqi = load_county_aqi_data(site_list)
    print(f"{aqi.columns=}")
    site_ids = aqi.unique(subset=["Defining Site"], maintain_order=True)
    print(f"{site_ids.select('Defining Site')=}")


def test_dbscan_variable(variable: str, start_date: str, end_date: str) -> (float, float):
    data = _get_data_for_variable(variable)
    data = data.drop_nulls()
    data = data.filter(pl.col("Date Local") >= start_date).filter(pl.col("Date Local") <= end_date)
    times = []
    for _i in range(NUM_TRIALS_PER_VARIABLE):
        start = time.time()
        detect_anomaly_dbscan(data, variable)
        end = time.time()
        times.append(end - start)
    return np.mean(times, dtype=np.float64), np.std(times, dtype=np.float64)


def test_dbscan_time(variables: list, start_date: str, end_date: str):
    print(f"Starting dbscan time tests for {start_date} to {end_date}...")
    means = []
    stds = []
    for variable in variables:
        mean, std = test_dbscan_variable(variable, start_date, end_date)
        means.append(mean)
        stds.append(std)
    results = zip(BENCHMARK_VARIABLES, means, stds)
    return results


def print_test_results(results, start_date: str, end_date: str):
    print(f"\nResults time tests for: {start_date} to {end_date}:")
    for variable, time_mean, time_std in results:
        print(
            f"Average time to run {variable} dbscan detection (rough CI): "
            f"{time_mean:.2f} +/- "
            f"{1.96 * time_std / np.sqrt(NUM_TRIALS_PER_VARIABLE):.4f} "
            f"seconds (standard deviation={time_std:.4f}).")


def test_bdscan_match_map_site(variable: str, site_id: str):
    # line chart data pattern
    site_data = _get_site_data(site_id)
    df = site_data[variable]
    df = df.drop_nulls(subset=variable)
    clustering = create_clusters_dbscan(df, variable)
    # map chart data pattern
    df = _get_data_for_variable(variable)
    df = df.filter(pl.col("site_id") == site_id)
    df = df.drop_nulls(subset=['Arithmetic Mean'])
    result = detect_anomaly_dbscan(df, variable)
    assert (-1 in clustering.labels_) == (result.select(pl.first("outlier")).item() == -1), f"Mismatch between map and line chart outlier detection for {variable=} at {site_id=}"


def main(timetest: bool = False):
    print("Basic functionality testing...")
    test_dbscan_syntax()

    # # ISSUE: between dates and county vs site issues, can't match model dev
    # print("\nImplementation testing, confirm outliers found in model "
    #       "development are also identified in visualization...")
    # site_list = ["01-001-0002", "04-003-2003", "08-001-3001", "09-003-1123"]
    # test_model_dev(site_list)

    # test speed Benchmark: 16.3 sec PM10, 39.8 sec AQI, 36.4PM2.5 (FRM)
    # On Abhi's machine DBSCAN on 2018-2024 data for AQI takes 5.5 seconds
    if timetest:
        start = "2018-01-01"
        end_1year = "2019-01-01"
        end_6year = "2024-01-01"
        results1 = test_dbscan_time(BENCHMARK_VARIABLES, start, end_1year)
        results6 = test_dbscan_time(BENCHMARK_VARIABLES, start, end_6year)
        print("\n---- 1 YEAR TEST ----")
        print_test_results(results1, start, end_1year)
        print("\n---- 6 YEAR TEST ----")
        print_test_results(results6, start, end_6year)


    # Confirm map + line chart match for outlier detection with DBSCAN
    # Example, now fixed: CO site 84115 red on map, no outlier on line chart
    variable = 'CO'
    sites = ["56910", "84115"]
    for site in sites:
        test_bdscan_match_map_site(variable, site)


    # print("\nTODO: Implement outlier identification accuracy testing after,"
    #       " parameter optimization and seasonal outlier detection added.")


if __name__ == "__main__":
    main(timetest=False)
