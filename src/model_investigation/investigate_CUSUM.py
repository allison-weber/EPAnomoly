import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

pd.set_option('display.max_columns', 10)


def load_county_aqi_data() -> pd.DataFrame:
    df = pd.read_csv("../data/Combined_AQI_By_County.part01/Combined_AQI_By_County.csv",
                     parse_dates=['Date'], dtype={'County Code': 'Int32'})
    df = df[['County Code', 'Date', 'AQI']]
    return df


def plot_county_aqi(data: pd.DataFrame, num_rows: int = 2, num_cols: int = 2):
    figure, axis = plt.subplots(num_rows, num_cols)
    for r in range(num_rows):
        for c in range(num_cols):
            county_num = r * num_rows + c + 1
            temp = data[data['County Code'] == county_num]
            axis[r, c].plot(temp['Date'], temp['AQI'])
            axis[r, c].set_title(f'AQI for County {county_num}')
    plt.show()


# TODO calc 4 all counties, for now just using 1 county as example
# TODO also detect decrease outliers
def calcCUSUM(df: pd.DataFrame, county_num: int = 1, c_std: float = 0.5,
              threshold_std: int = 4) -> pd.DataFrame:
    means = df.groupby('County Code').mean().rename(columns={'AQI': 'meanAQI'})
    std_devs = df.groupby('County Code').std().rename(columns={'AQI': 'stdAQI'})
    df = pd.merge(df, means, on='County Code', how='inner')

    data = df[df['County Code'] == county_num].reset_index(drop=True)
    c_value = std_devs.loc[county_num, 'stdAQI'] * c_std
    mean = means.loc[county_num, 'meanAQI']
    upper_threshold = mean + threshold_std * c_value

    data['cusum_increase'] = data['meanAQI']    # temp value
    for i in range(1, len(data)):
        data.loc[i, 'cusum_increase'] = max(data.loc[i - 1, 'cusum_increase'] +
                                         (data.loc[i, 'AQI'] - data.loc[i, 'meanAQI']) -
                                         c_value, 0)
    data['cusum_increase_outlier'] = data['cusum_increase'] >= upper_threshold
    plot_cusum(data, county_num, mean, c_value, upper_threshold)
    return data

    # might be more elegant solution with shift
    # data['prevAQI'] = data.groupby('County Code')['AQI'].shift().fillna(data['meanAQI'])


def plot_cusum(data: pd.DataFrame, county_num: int, mean: float,
               c_value: float, upper_threshold: float):
    # plot with outliers different color
    upper = np.ma.masked_where(data['cusum_increase_outlier'], data['AQI'])
    normal = np.ma.masked_where(data['cusum_increase_outlier'] == False, data['AQI'])
    fig, ax = plt.subplots()
    ax.plot(data['Date'], upper, data['Date'], normal)
    plt.title(f"County {county_num}: "
              f"mean={round(mean, 1)}, "
              f"c={round(c_value,1)}, threshold={round(upper_threshold, 1)}")
    plt.show()


def main():
    # Warning: code is slow, NOT optimized for speed
    start = time.time()
    df = load_county_aqi_data()
    plot_county_aqi(df, 2, 2)
    outliers = calcCUSUM(df, 3, 2, 6)
    stop = time.time()
    # print(f"time using pandas: {stop-start}")


if __name__ == '__main__':
    main()