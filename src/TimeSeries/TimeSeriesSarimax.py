from src.TimeSeries.TimeSeriesForecast import TimeSeriesForecast
import statsmodels.api as sm
from typing import Tuple
import numpy as np
import pandas as pd


class TimeSeriesSarimax(TimeSeriesForecast):
    """
    Class that inherits from TimeSeriesForecast Class. It handles forecasting of future values by using Sarimax.
    """

    def __init__(self):
        super().__init__()

    def fit(self, name: str, order: tuple, seasonal_order: tuple):
        """
        Fits a SARIMAX model to {name} temporal series given the order and seasonal_order parameters of the
        statsmodels.tsa.statespace.SARIMAX constructor.

        :param name:                        Name of temporal serie.
        :param order:                       Order parameter of SARIMAX.
        :param seasonal_order:              Seasonal order of SARIMAX.
        """
        mod = sm.tsa.statespace.SARIMAX(self[name], order=order, seasonal_order=seasonal_order,
                                        enforce_stationarity=False, enforce_invertibility=False)
        results = mod.fit(maxiter=200, disp=False)
        self._models[name] = mod
        self._results[name] = results
        print(results.summary())

    def _proxy_predict(self, name: str, start: str, end: str) -> Tuple[pd.Series, np.ndarray]:
        last_result = self._results[name]
        pred = last_result.get_prediction(start=start, end=end)
        return pred.predicted_mean, pred.conf_int()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = TimeSeriesSarimax()
    t.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv", index_col='Month')
    order = (1, 2, 2)
    seasonal_order = (2, 2, 2, 12)
    name = 'Red '
    t.fit(name, order, seasonal_order)
    insample_mean, insample_interval = t.predict_in_sample(name)

    fig, ax = t.plot_forecast(name, start='1993-01-01', end='1995-02-01', forecast_label='Forecast')
    t.plot_serie(name, ax, start='1992-01-01')
    plt.show()