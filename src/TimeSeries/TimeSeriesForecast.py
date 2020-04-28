from src.TimeSeries.TimeSeries import TimeSeries
import statsmodels.api as sm
import matplotlib.pyplot as plt


class ResultsNotFound(Exception):
    def __init__(self, name: str):
        super(ResultsNotFound, self).__init__(f"There are not SARIMAX models fitted for: {name}. Please, fit before"
                                              f"predict.")


class TimeSeriesForecast(TimeSeries):
    # TODO: Add docs
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
        # TODO: Add docs
        pred_mean, pred_ci = self.predict_in_sample(name)
        y_truth = self[name]
        mse = ((pred_mean - y_truth) ** 2).mean()
        print(f"Reported MSE for last SARIMAX model trained in {name}: {mse}")

    def predict_in_sample(self, name: str):
        """
        Does in-sample prediction of specified serie.
        :param name:                            Name of the serie.
        :return:                                A PredictionResultsWrapper instance
                                                (see more references on statsmodels docs).
        """

        start = self._data.index[0]
        end = self._data.index[-1]
        return self._predict(name, start, end)

    def predict_out_of_sample(self, name: str, start: str, end: str):
        """

        Does a out of sample prediction using last model fitted for {name} serie.

        :param name:                          Name of the serie.
        :param start:                         String representation of a date from which predict.
        :param end:                           String representation of a date until which predict.
        """
        return self._predict(name=name, start=start, end=end)

    def _predict(self, name: str, start: str, end: str):
        # TODO: Add docs
        if name not in self._results.keys():
            raise ResultsNotFound(name)

        last_result = self._results[name]
        pred = last_result.get_prediction(start=start, end=end)
        return pred.predicted_mean, pred.conf_int()

    def plot_insample_pred(self, name: str):
        # TODO: Add docs
        self.plot_forecast(name, start=self[name].index[0], end=self[name].index[-1],
                           forecast_label='In-sample forecast')

    def plot_forecast(self, name: str, start: str, end: str, forecast_label: str):
        # TODO: Add docs
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




