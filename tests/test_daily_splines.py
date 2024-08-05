import numpy as np
import polars as pl
import time
import sys
sys.path.append("..") # So we can get the functions from viz/

from viz.bspline_daily_outliers import detect_anomalies_bsplines_daily

BENCHMARK_VARIABLES = [
    "AQI",
    "PM2.5 FRM",
    "PM2.5 non-FRM",
    "PM10",
    "CO"
]

CRITICAL_VALUE = 6 # Doesn't really matter for the tests, going with the classic 6-sigma
NUM_TRIALS_PER_VARIABLE = 10

def _get_data_for_variable(variable: str):
    """Returns combined.parquet for the given variable."""
    return pl.read_parquet(f"../src/data/daily/{variable}/combined.parquet").sort("Date Local")

def main():
    for variable in BENCHMARK_VARIABLES:
        data = _get_data_for_variable(variable)
        variable_times = []
        for _i in range(NUM_TRIALS_PER_VARIABLE):
            pass
            start = time.time()
            _ = detect_anomalies_bsplines_daily(data, variable, CRITICAL_VALUE)
            end = time.time()
            variable_times.append(end-start)
        variable_times = np.array(variable_times)
        print(f"Average time to run {variable} daily spline detection (rough CI): {np.mean(variable_times):.2f} +/- {1.96 * np.std(variable_times) / np.sqrt(NUM_TRIALS_PER_VARIABLE):.4f} seconds.")

if __name__ == "__main__":
    main()