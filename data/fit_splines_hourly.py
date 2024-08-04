from multiprocessing import Pool, cpu_count
import numpy as np
import polars as pl
import time
from tqdm import tqdm

from glob import glob
from pathlib import Path
from scipy.interpolate import BSpline
from sklearn.metrics import mean_squared_error

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


def fit_bsplines_hourly(data: pl.DataFrame, value_name: str) -> pl.DataFrame:
    """
    Read in an hourly dataframe and fit a bspline to the data for each day.

    :param data: hourly data for a single site
    :param value_name: name of the column to fit the bspline to
    """
    data = data.with_columns([(pl.col('Date Local').str.strptime(pl.Datetime, "%Y-%m-%d"))])
    all_dates = data['Date Local'].unique().sort()

    # Calculate Bspline Basis for a day of full data
    xx = np.linspace(0, 1, 24)
    knots = np.linspace(0, 1, 6)
    deg = 3
    B = BSplineBasis(xx, knots, deg)[:,:-2]

    mse_dict = {
        "date": [],
        "hourly_spline_mse": []
    }

    for date in all_dates:
        mse_dict["date"] += [date.strftime("%Y-%m-%d")]
        filtered_df = data.filter(pl.col("Date Local") == date)
        y = np.array(filtered_df[value_name])
        
        try:
            if len(filtered_df) == 24:
                yhat = B @ np.linalg.pinv(B) @ y
            else:
                # Create a new basis and domain with the correct number of points for days with missing data
                xx = np.linspace(0, 1, len(filtered_df))
                num_knots = min(5, len(filtered_df) // 2)
                knots_smaller = np.linspace(0, 1, num_knots)
                B_smaller = BSplineBasis(xx, knots_smaller, deg)[:,:-2]
                yhat = B_smaller @ np.linalg.pinv(B_smaller) @ y

            mse = mean_squared_error(y, yhat)
            mse_dict["hourly_spline_mse"] += [mse]

        except Exception as e:
            # Sometimes the data are missing for a day, so we can't fit a bspline
            # or the data are too sparse to fit a bspline
            mse_dict["hourly_spline_mse"] += [None]
            continue
    
    return pl.DataFrame(mse_dict)

def fit_splines_for_site(site_id: int, export: bool = True):
    """
    Fit bsplines to all variables for a site and compute the MSE for each day.

    :param site_id: id of the site to fit bsplines for
    :param export: whether to write the bspline MSEs to the daily parquet files.
    """
    # Find all variables for the site
    variables = glob(f"./hourly/sites/{site_id}/*.parquet")
    variables = [Path(x).stem for x in variables]

    for var in variables:
        # Check if both the hourly and daily data exists for the variable
        hourly_data_exists = Path(f"./hourly/sites/{site_id}/{var}.parquet").exists()
        daily_data_exists = Path(f"./daily/sites/{site_id}/{var}.parquet").exists()

        if hourly_data_exists and daily_data_exists:
            # Load hourly data for the site
            hourly_df = pl.read_parquet(f"./hourly/sites/{site_id}/{var}.parquet")
       
            daily_df = pl.read_parquet(f"./daily/sites/{site_id}/{var}.parquet")
            if "hourly_spline_mse" in daily_df.columns:
                daily_df = daily_df.drop("hourly_spline_mse")

            mse_df = fit_bsplines_hourly(hourly_df, value_name=var)
            # mse_df.write_parquet(f"./hourly/sites/{site_id}/{var}_bsplines.parquet")

            if export:
                daily_df.join(mse_df, left_on="Date Local", right_on="date", how="left")\
                    .write_parquet(f"./daily/sites/{site_id}/{var}.parquet")

def compute_bsplines(multicore: bool = True):
    """
    Fit bsplines for all hourly data for all sites and variables.

    :param multicore: whether to use multiprocessing to fit bsplines for all sites in parallel.

    Saves the bspline MSEs to the daily parquet files under the column name "hourly_spline_mse".
    """
    all_sites = glob("./hourly/sites/*")
    all_sites = [int(Path(x).stem) for x in all_sites]
    print(f"Number of sites: {len(all_sites)}")

    t0 = time.time()
    if multicore:
        with Pool(processes=cpu_count() - 2) as p:
            p.map(fit_splines_for_site, all_sites)
    else:
        for site_id in tqdm(all_sites): # log with a progress bar
            fit_splines_for_site(site_id)
    t1 = time.time()

    print(f"Time taken: {t1 - t0:.2f} seconds")
        
if __name__ == "__main__":
    # Takes about 20 min on my machine (Abhi)
    compute_bsplines(multicore=True)