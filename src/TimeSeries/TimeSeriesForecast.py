import matplotlib.pyplot as plt
import statsmodels.api as sm

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
        super(TimeSeries, self).__init__()
        self._models = {}
        self._results = {}

    def fit_sarimax(self, name: str, order: tuple, seasonal_order: tuple):
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

        last_result = self._results[name]
        pred = last_result.get_prediction(start=start, end=end)
        return pred.predicted_mean, pred.conf_int()

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


if __name__ == '__main__':
    t = TimeSeriesForecast()
    t.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv", index_col='Month')
    order = (1, 2, 2)
    seasonal_order = (2, 2, 2, 12)
    name = 'Red '
    t.fit_sarimax(name=name, order=order, seasonal_order=seasonal_order)
    insample_mean, insample_interval = t.predict_in_sample(name)

    fig, ax = t.plot_forecast(name, start='1993-01-01', end='1995-02-01', forecast_label='Forecast')
    t.plot_serie(name, ax, start='1992-01-01')
    plt.show()
