import numpy as np
import polars as pl
from tqdm import tqdm

from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
from scipy.interpolate import BSpline

def BSplineBasis(x: np.array, knots: np.array, degree: int) -> np.array:
    #B spline function
    nKnots = knots.shape[0]
    lo = min(x[0], knots[0])
    hi = max(x[-1], knots[-1])
    augmented_knots = np.append(np.append([lo]*degree, knots), [hi]*degree)
    DOF = nKnots + degree +1 
    spline = BSpline(augmented_knots, np.eye(DOF), degree, extrapolate=False)
    B = spline(x)
    return B

def fit_spline_daily(data: pl.DataFrame, value_name: str = "Arithmetic Mean", num_knots: int = None) -> pl.DataFrame:
    """
    Read in an hourly dataframe and fit a bspline to the data for each day.

    :param data: hourly data for a single site
    :param value_name: name of the column to fit the bspline to
    """
    if num_knots is None:
        num_knots = len(data) // 4

    # Calculate Bspline Basis for the whole dataset
    xx = np.linspace(0, 1, len(data))
    knots = np.linspace(0, 1, num_knots)
    deg = 3
    try:
        B = BSplineBasis(xx, knots, deg)[:,:-2]
    except (IndexError, ValueError): # Happens when there is no data
        return_col = np.array([np.nan] * len(data))
        return data.with_columns(rmse_daily_spline = return_col)

    # Fit the spline
    try:
        y = np.array(data[value_name])
        yhat = B @ np.linalg.pinv(B) @ y
        rmse = np.sqrt((y - yhat) ** 2)
    except Exception:
        rmse = np.array([np.nan] * len(data))

    return data.with_columns(rmse_daily_spline = rmse)

def fit_splines_for_site(site_id: int, export: bool = True):
    """
    Fit bsplines to all variables for a site and compute the MSE for each day.

    :param site_id: id of the site to fit bsplines for
    :param export: whether to write the bspline MSEs to the daily parquet files.
    """
    # Find all variables for the site
    variables = glob(f"./daily/sites/{site_id}/*.parquet")
    variables = [Path(x).stem for x in variables]

    for var in variables:
        # Load the daily data and dedupe (mostly for VOCs)
        df = pl.read_parquet(f"./daily/sites/{site_id}/{var}.parquet")\
            .unique("Date Local", keep="first")

        # Drop the rmse_daily_spline column if it already exists
        if "rmse_daily_spline" in df.columns:
            df = df.drop("rmse_daily_spline")

        mse_df = fit_spline_daily(df, value_name=var)

        if export:
            mse_df.write_parquet(f"./daily/sites/{site_id}/{var}.parquet")

def fit_daily_bsplines(multicore: bool = False):
    all_sites = glob("./daily/sites/*")
    all_sites = [int(Path(x).stem) for x in all_sites]

    if multicore:
        num_processes = cpu_count() - 2 if cpu_count() > 4 else cpu_count()
        with Pool(processes=num_processes) as p:
            p.map(fit_splines_for_site, all_sites)
    else:
        for site in tqdm(all_sites):
            fit_splines_for_site(site)

def main():
    fit_daily_bsplines(multicore=False)

if __name__ == "__main__":
    main()