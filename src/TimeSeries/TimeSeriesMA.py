from src.TimeSeries.TimeSeriesForecast import TimeSeriesForecast
import statsmodels.api as sm
from typing import Tuple
import pandas as pd


class TimeSeriesMA(TimeSeriesForecast):
    """
    Class that inherits from TimeSeriesForecast Class. It handles forecasting of future values by using Moving Average.
    """

    def __init__(self):
        super().__init__()

    def fit(self, name: str, *,  order: int = 12):
        self._models[name] = f"MA({order})_{name}"
        self._results[name] = self[name].rolling(order).mean().dropna()

    def _proxy_predict(self, name: str, start: str, end: str) -> Tuple[pd.Series, pd.DataFrame]:
        last_result = self._results[name]
        return last_result, pd.DataFrame()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = TimeSeriesMA()
    t.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv", index_col='Month')
    name = 'Rose '
    fig, ax = plt.subplots()
    t.difference(interval=12)
    t[name].plot(ax=ax, label='after first diff')
    t.difference(interval=1)
    t[name].plot(ax=ax, label='after second diff')
    plt.legend()
    plt.show()
    # t.fit(name, 12)
    # insample_mean, insample_interval = t.predict_in_sample(name)
    #
    # fig, ax = t.plot_forecast(name, start='1993-01-01', end='1995-02-01', forecast_label='Forecast')
    # t.plot_serie(name, ax, start='1992-01-01')
    # plt.show()
