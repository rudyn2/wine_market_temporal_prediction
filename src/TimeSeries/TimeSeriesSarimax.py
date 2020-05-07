from typing import Tuple

import pandas as pd
import statsmodels.api as sm

from src.TimeSeries.TimeSeriesForecast import TimeSeriesForecast
from src.Utils.Utils import Utils


class TimeSeriesSarimax(TimeSeriesForecast):
    """
    Class that inherits from TimeSeriesForecast Class. It handles forecasting of future values by using Sarimax.
    """

    def __init__(self):
        super().__init__()
        self._params_file_paths: dict = {}

    def load_params_file(self, name: str, params_path: str) -> None:
        """
        load the path to params tried file
        """
        self._params_file_paths[name] = params_path

    def fit_on_best(self, name: str):
        """
        Fits a SARIMA model with the best params found in params file given by AIC.
        """
        best_params, _ = Utils.read_params_aic(self._params_file_paths[name], best=True)
        self.fit(name, order=best_params[:3], seasonal_order=best_params[3:])

    def fit(self, name: str, *, order: tuple = (1, 0, 0), seasonal_order: tuple = (1, 0, 0, 12)):
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

    def _proxy_predict(self, name: str, start: str, end: str) -> Tuple[pd.Series, pd.DataFrame]:
        last_result = self._results[name]
        pred = last_result.get_prediction(start=start, end=end)
        return pred.predicted_mean, pred.conf_int()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    repo_path = Utils.get_repo_path()
    name = 'Red '

    t = TimeSeriesSarimax()
    t.load(os.path.join(repo_path, 'data/AustralianWines.csv'), index_col='Month')

    t.fit(name, order=(1, 2, 2), seasonal_order=(2, 2, 2, 12))

    fig, ax = t.plot_forecast(name, start='1993-01-01', end='1995-02-01', forecast_label='Forecast')
    t.plot_serie(name, ax, start='1992-01-01')
    plt.show()
