import pandas as pd
import numpy as np
from DBSCAN_test import load_county_aqi_data_pl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import pmdarima as pm
from statsmodels.tsa.seasonal import STL


def getARIMAdiff(county: pd.DataFrame):
    p_value = 1
    AQIdiff = county['AQI']
    q = 0   # order of differencing for ARIMA
    while p_value >= 0.05:
        ADF_result = adfuller(AQIdiff)
        print(f'ADF Statistic: {ADF_result[0]}') # more negative => stronge rejection null hypothesis
        p_value = ADF_result[1]
        print(f'{p_value=}')
        plot_acf(AQIdiff, lags=20)
        plt.show()

        if p_value < 0.05:
            print("Reject null hypothesis => time series is stationary")
        else:
            print("Fail to reject null hypothesis => time series is not stationary")
            AQIdiff = np.diff(AQIdiff, n=1)
            q += 1
    return q, AQIdiff


# based on ARIMA book: https://learning.oreilly.com/library/view/time-series-forecasting/9781617299889
def investigateARIMA(county: pd.DataFrame, county_num: int = 1):
    # check if time series stationary- time consuming
    # County 1, 3, 4 = stationary, County 2, =  non-
    # NOTE:  If both the ACF and PACF plots exhibit a slow decay or sinusoidal
    # pattern => no order for the MA(q) or AR(p) process can be inferred
    [q, AQIdiff] = getARIMAdiff(county)
    plot_pacf(AQIdiff, lags=20)
    plt.show()
    print('END')


if __name__ == '__main__':
    COUNTY = 1  # For demo purposes just running 1 county at a time
    SHOW_ARIMA_STEPS = False

    df = load_county_aqi_data_pl()
    df_county = df[df['County Code'] == COUNTY].dropna().reset_index(drop=True)
    print(f"data loaded for County {COUNTY}")
    if SHOW_ARIMA_STEPS:
        investigateARIMA(df_county)

    # https://pypi.org/project/pmdarima/
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html#statsmodels.tsa.seasonal.STL
    # TODO fill missing rather than remove as above
    # TODO not working right
    model = pm.auto_arima(df_county['AQI'], seasonal=True, m=12)
    print(model)
    res = STL(df_county['AQI']).fit(period=12)
    res.plot()
    plt.show()
