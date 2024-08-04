import polars as pl

from glob import glob
from pathlib import Path


def _get_site_data(site_id: str) -> dict[str, pl.DataFrame]:
    """
    Gets all of the data for the given site ID.
    """
    available_variables = glob(f"../data/daily/sites/{site_id}/*.parquet")
    available_variable_options = [Path(x).stem for x in available_variables]

    data_dict = {}

    for filename, varname in zip(available_variables, available_variable_options):
        data_dict[varname] = pl.read_parquet(filename).sort("Date Local")

    # Sort data_dict by alphabetical order of the keys
    data_dict = dict(sorted(data_dict.items()))

    return data_dict

def _get_data_for_variable(variable: str):
    """
    Loads combined.parquet for the given variable.
    """
    return pl.read_parquet(f"../data/daily/{variable}/combined.parquet").sort("Date Local")

def main():
    # For testing utilities, none of this code is loaded in when the script is imported
    data = _get_site_data("4710")
    print(data)

if __name__ == "__main__":
    main()