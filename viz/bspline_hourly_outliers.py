import polars as pl
import time

MIN_POINTS_SPLINES = 20

def find_site_outliers_hourly_spline_mse(data: pl.DataFrame, site_id: str, critical_value = 3.5) -> dict:
    """
    Find outliers using data for a single site, only if the site has at least MIN_POINTS_SPLINES data points.
    :param data: data for a single site
    :param critical_value: Z-score threshold for outlier detection

    :return: dictionary with keys site_id[str], outlier[int], and Hourly spline anomaly detected?[str]
    """
    # set initial result to no-outlier found (0)
    result = {"site_id": site_id, "outlier": 0, "Hourly spline anomaly detected?": 'No'}

    if data.shape[0] <= MIN_POINTS_SPLINES or "hourly_spline_mse" not in data.columns:
        result["Hourly spline anomaly detected?"] = "Insufficient data"
        return result, pl.DataFrame()
    
    # Transform mse column into z scores
    avg_mse = data["hourly_spline_mse"].mean()
    std_mse = data["hourly_spline_mse"].std()
    data = data.with_columns(((data["hourly_spline_mse"] - avg_mse) / std_mse).alias("zscore"))

    # Find outliers using the Z score
    data = data.with_columns(pl.when(pl.col("zscore") > critical_value)
                                .then(pl.lit(1))
                                .otherwise(pl.lit(0))
                                .alias("outlier"))
    
    # If any outliers are found, update the result
    if data["outlier"].sum() > 0:
        result["outlier"] = 1
        result["Hourly spline anomaly detected?"] = 'Yes'

    return result, data

def detect_anomalies_bsplines_hourly(data: pl.DataFrame,
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
    Columns: site_id, outlier, Hourly spline anomaly detected?
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
        result, _ = find_site_outliers_hourly_spline_mse(site_data, site_id, critical_value)
        results.append(result)

    df = pl.DataFrame(results)
    return df

def main():
    from viz_utils import _get_data_for_variable
    data = _get_data_for_variable("SO2")

    data = data.drop_nulls()

    start_date = data["Date Local"].min()
    end_date = data["Date Local"].max()

    assert start_date <= end_date, "Start date must be before end date."

    test = detect_anomalies_bsplines_hourly(data, "SO2", 25)
    print(test)

    gb = test.group_by("Hourly spline anomaly detected?").agg(pl.count("site_id").alias("count"))
    print(gb)


if __name__ == "__main__":
    main()