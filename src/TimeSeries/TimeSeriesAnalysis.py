import pprint
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.TimeSeries.TimeSeries import TimeSeries


# TODO: Check if package indexing is generalizable


class TimeSeriesAnalysis(TimeSeries):
    """
    Class that inherits from TimeSeries Class. It handles the analysis of a time series by performing
    a number of test that give a notion on how stationary the series is.
    """

    def __init__(self):
        super().__init__()
        self.__stats = {}

    def stats(self, name: str, verbose: bool = True):
        """
        It triggers the stats computing and its display
        :param name:                    Name of particular series that wants to be analyzed.
        :param verbose:                 If true the stats will be printed.
        """
        assert name in self._col_names, f"There isn't a column called {name}."
        self.__stats[name] = {}
        self._compute_stats(name)
        if verbose:
            pprint.pprint(self.__stats[name])

    def mse(self, name: str, external_series: pd.Series):
        """
        Computes the mean square error (MSE) between the time series indexed by name and an external time series
        given in external_series, by taking only the points where their indexes match. It's a similarity measure
        in someway.

        :param name:                    Name of the temporal series.
        :param external_series:         External series passed for comparison.
        :return:                        MSE between the tow series.
        """
        intersection_index = self._data.loc[external_series.index].index
        true_values = self[name].loc[intersection_index]
        return ((external_series - true_values) ** 2).mean()

    def _compute_stats(self, name: str):
        """
        Compute all the relevant stats.
        :param name:                    Name of the temporal series.
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="p-value is greater than the indicated p-value")
            warnings.filterwarnings("ignore", message="p-value is smaller than the indicated p-value")
            self._compute_adfuller(name)
            self._compute_kpss(name)
            self._compute_kurtosis(name)

    def _compute_adfuller(self, name: str):
        """
        Performs the Augmented Dickey-Fuller unit root test for stationarity and store the main results
        for further display.

        :param name:                    Name of the temporal series.
        """
        adf_stat, p_value_adf, _, _, critical_vals_adf, _ = adfuller(self[name], regression='ct')
        stats = {
            'ADF Statistic': adf_stat,
            'p-value': p_value_adf,
            'Critical values': critical_vals_adf
        }
        self.__stats[name]['Dickey-Fuller'] = stats

    def _compute_kpss(self, name: str):
        """
        Performs the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity and store the main results
        for further display.

        :param name:                    Name of the temporal series.
        """
        kpss_stat, p_value_kpss, _, critical_vals_kpss = kpss(self[name], regression='ct', nlags='auto')

        stats = {
            'KPSS Statistic': kpss_stat,
            'p-value': p_value_kpss,
            'Critical values': critical_vals_kpss
        }
        self.__stats[name]['KPSS'] = stats

    def _compute_kurtosis(self, name: str):
        """
        Calculates the kurtosis of the time series and store the statistic for further display. Kurtosis is
         a peakedness measurement of the time series distribution.

        :param name:                    Name of the temporal series.
        """
        stats = {
            'Kurtosis Statistic': self[name].kurtosis()
        }
        self.__stats[name]['Kurtosis'] = stats

    def plot_hist(self, name: str, ax: plt.Axes = None, bins: int = 10):
        """
        Plots an histogram.
        """
        if ax is None:
            self[name].hist(bins=bins)
        self[name].hist(bins=bins, ax=ax)

    @staticmethod
    def _is_rejected(p_value, p_thresh: float = 0.05):
        """
        Checks if a null hypothesis is rejected or not.
        :param p_value:                     p_value of the test.
        :param p_thresh:                    significance level (usual values are 0.05 or 0.01)
        :return:                            True if the test is rejected.
        """
        if p_value < p_thresh:
            return True
        return False

    def report_stationary(self, verbose: bool = False):
        """
        It uses the p-value statistic to check if stationary is achieved using KPSS and ADF statistical tests. The rules
        that dictates if a series is stationary or not are the following.

        1) Both fails to reject --> not stationary
        2) Both reject --> stationary
        3) reject KPSS and fails to reject ADF --> trend stationary
        4) fails to reject KPSS and reject ADF --> difference stationary
        """
        for ts_name in self._col_names:

            print(f"Testing: {ts_name}")
            self.stats(ts_name, verbose=verbose)
            kpss_p = self.__stats[ts_name]['KPSS']['p-value']
            adf_p = self.__stats[ts_name]['Dickey-Fuller']['p-value']

            # checks the statistical tests
            if self._is_rejected(adf_p):
                result = 'stationary'
            else:
                # not enough evidence -> an additional test is required
                if self._is_rejected(kpss_p):
                    result = 'not stationary'
                else:
                    result = 'trend stationary'
            print(f"{ts_name}: {result}\n")


if __name__ == '__main__':
    t = TimeSeriesAnalysis()
    t.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv", index_col='Month')
    t.report_stationary(verbose=False)
    t.difference(interval=1)
    t.report_stationary(verbose=True)
