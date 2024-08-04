import polars as pl
import time

MIN_POINTS_SPLINES = 20

def find_site_outliers_daily_spline_error(data: pl.DataFrame, site_id: str, critical_value = 3.5) -> dict:
    """
    Find outliers using data for a single site, only if the site has at least MIN_POINTS_SPLINES data points.
    :param data: data for a single site
    :param critical_value: Z-score threshold for outlier detection

    :return: dictionary with keys site_id[str], outlier[int], and Daily spline anomaly detected?[str]
    """
    # set initial result to no-outlier found (0)
    result = {"site_id": site_id, "outlier": 0, "Daily spline anomaly detected?": 'No'}

    if data.shape[0] <= MIN_POINTS_SPLINES or "rmse_daily_spline" not in data.columns:
        result["Daily spline anomaly detected?"] = "Insufficient data"
        return result, pl.DataFrame()
    
    # Transform mse column into z scores
    avg_mse = data["rmse_daily_spline"].drop_nulls().mean()
    std_mse = data["rmse_daily_spline"].drop_nulls().std()
    data = data.with_columns(((pl.col("rmse_daily_spline") - avg_mse) / std_mse).alias("zscore"))

    # Find outliers using the Z score
    data = data.with_columns(
        pl.when(
            (pl.col("zscore") > critical_value) & (pl.col("zscore").is_not_null()) & (pl.col("zscore").is_not_nan())
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("outlier")
    )
    
    # If any outliers are found, update the result
    if data["outlier"].sum() > 0:
        result["outlier"] = 1
        result["Daily spline anomaly detected?"] = 'Yes'

    return result, data

def detect_anomalies_bsplines_daily(data: pl.DataFrame,
                            variable: str,
                            critical_value: float = 3.5,
                            start_date: str = None,
                            end_date: str = None) -> pl.DataFrame:
    """
    Basic implementation of B-Spline outlier detection, 0=No anomaly, 1=anomaly detected
    :param data: dataframe with data for all sites
    :param variable: air quality measure evaluating for anomalies
    :param critical_value: Z-score threshold for outlier detection
    :return: dataframe with site ids and anomaly detection indicator column.
    Columns: site_id, outlier, Daily spline anomaly detected?
    """
    # Determine the sites with the given variable

    available_sites = list(data.select("site_id").unique().to_pandas()["site_id"])

    # Create a list of dataframes for each site
    all_dfs = {site_id: pl.read_parquet(f"../data/daily/sites/{site_id}/{variable}.parquet") for site_id in available_sites}
    results = []

    for site_id, site_data in all_dfs.items():
        if start_date is not None:
            site_data = site_data.filter(pl.col("Date Local") >= start_date)
        if end_date is not None:
            site_data = site_data.filter(pl.col("Date Local") <= end_date)

        result, _ = find_site_outliers_daily_spline_error(site_data, site_id, critical_value)
        results.append(result)

    return pl.DataFrame(results)

def main():
    from viz_utils import _get_data_for_variable
    
    data = _get_data_for_variable("SO2")
    start = time.time()
    df = detect_anomalies_bsplines_daily(data, "SO2", 15)
    print(df)
    print(f"Time taken: {time.time() - start}")


    TEST_SITE = 38153
    TEST_VAR="AQI"
    site_data = pl.read_parquet(f"../data/daily/sites/{TEST_SITE}/{TEST_VAR}.parquet")
    result, data = find_site_outliers_daily_spline_error(site_data, TEST_SITE, 10)

    print(result)
    print(data.sort("zscore"))

if __name__ == "__main__":
    main()