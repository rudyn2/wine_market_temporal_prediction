import abc
from typing import Tuple

import pandas as pd

from src.TimeSeries.TimeSeries import TimeSeries


class ResultsNotFound(Exception):
    def __init__(self, name: str):
        super(ResultsNotFound, self).__init__(f"There are not SARIMAX models fitted for: {name}. Please, fit before"
                                              f"predict.")


class TimeSeriesForecast(TimeSeries):
    """
    Class that inherits from TimeSeries Class. It handles forecasting of future values by using the known values of
    the series and models, like SARIMA. It also has some methods for visualization of forecasting results.
    """

    def __init__(self):
        super().__init__()
        self._models = {}
        self._results = {}

    def get_mse(self, name):
        """
        Computes de Mean Square Error of the original time series indexed by {name} and the prediction
        of the fitted model for the same series. It's a measure of how well the model fits to the original data.

        :param name:                        Name of temporal serie.
        """
        pred_mean, pred_ci = self.predict_in_sample(name)
        y_truth = self[name]
        mse = ((pred_mean - y_truth) ** 2).mean()
        print(f"Reported MSE for last SARIMAX model trained in {name}: {mse}")

    def predict_in_sample(self, name: str):
        """
        Does in-sample prediction of specified series.
        :param name:                            Name of the series.
        :return:                                A PredictionResultsWrapper instance
                                                (see more references on statsmodels docs).
        """

        start = self._data.index[0]
        end = self._data.index[-1]
        return self._predict(name, start, end)

    def predict_out_of_sample(self, name: str, start: str, end: str):
        """
        Does a out of sample prediction using last model fitted for {name} series. Forecasting.

        :param name:                          Name of the serie.
        :param start:                         String representation of a date from which predict.
        :param end:                           String representation of a date until which predict.
        """
        return self._predict(name=name, start=start, end=end)

    def _predict(self, name: str, start: str, end: str):
        """
        It either do in-sample prediction or out of sample forecasting, depending if the {start} and {end} range falls
        in or out of the time range of the data used for fitting the model.

        :param name:                        Name of temporal serie.
        :param start:                       start of forecast or in-sample
        :param end:                         end of forecast or in-sample
        :return:                            tuple of two arrays
                                                1) mean of the prediction for each point in range
                                                2) 95% confidence interval for each point in range
        """
        if name not in self._results.keys():
            raise ResultsNotFound(name)

        return self._proxy_predict(name, start, end)

    def plot_insample_pred(self, name: str):
        """
        Plot the fitted model prediction for the data's original time range, data indexed by {name}.
        :param name:                        Name of temporal serie.
        """
        self.plot_forecast(name, start=self[name].index[0], end=self[name].index[-1],
                           forecast_label='In-sample forecast')

    def plot_forecast(self, name: str, start: str, end: str, forecast_label: str):
        """
        PLot the predicted data indexed by {name} in the time range specified by {start} and {end}, it plots
        the mean value predicted alongside the confidence interval.

        :param name:                        Name of temporal serie.
        :param start:                       Start of temporal axis
        :param end:                         End of temporal axis
        :param forecast_label:              label for the time series prediction
        """
        fig, ax = self._get_customized_figure()
        pred_mean, pred_ci = self._predict(name, start=start, end=end)
        pred_mean.plot(ax=ax, label=forecast_label, alpha=.7)
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel(name)
        ax.legend()
        return fig, ax

    @abc.abstractmethod
    def _proxy_predict(self, result, start: str, end: str) -> Tuple[pd.Series, pd.DataFrame]:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, name, **kwargs):
        raise NotImplementedError
