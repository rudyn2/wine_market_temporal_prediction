from src.TimeSeries.TimeSeriesForecast import TimeSeriesForecast
import statsmodels.api as sm
from typing import Tuple
import numpy as np
import pandas as pd


class TimeSeriesMA(TimeSeriesForecast):
    """
    Class that inherits from TimeSeriesForecast Class. It handles forecasting of future values by using Moving Average.
    """

    def __init__(self):
        super().__init__()

    def fit(self, name: str, order: int):
        model = sm.tsa.ARMA(self[name], order=(0, order))
        results = model.fit(maxiter=200, disp=False)
        self._models[name] = model
        self._results[name] = results

    def _proxy_predict(self, name: str, start: str, end: str) -> Tuple[pd.Series, np.ndarray]:
        last_result = self._results[name]
        predicted = last_result.predict(start=start, end=end)
        return predicted, None


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = TimeSeriesMA()
    t.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv", index_col='Month')
    order = (1, 2, 2)
    seasonal_order = (2, 2, 2, 12)
    name = 'Red '
    t.fit(name, 12)
    insample_mean, insample_interval = t.predict_in_sample(name)

    fig, ax = t.plot_forecast(name, start='1993-01-01', end='1995-02-01', forecast_label='Forecast')
    t.plot_serie(name, ax, start='1992-01-01')
    plt.show()
