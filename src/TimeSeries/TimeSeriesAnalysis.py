import pprint
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss

from src.TimeSeries.TimeSeries import TimeSeries


# TODO: Check if package indexing is generalizable


class TimeSeriesAnalysis(TimeSeries):

    def __init__(self):
        super().__init__()
        self.__stats = {}

    def stats(self, name: str):
        assert name in self._col_names, f"There isn't a column called {name}."
        self.__stats[name] = {}
        self._compute_stats(name)
        pprint.pprint(self.__stats[name])

    def mse(self, name: str, external_series: pd.Series):
        intersection_index = self._data.loc[external_series.index].index
        true_values = self[name].loc[intersection_index]
        return ((external_series - true_values)**2).mean()

    def _compute_stats(self, name: str):
        """
        Compute all the relevant stats.

        :param name:                    Name of the temporal serie.
        :return:
        """

        self._compute_adfuller(name)
        self._compute_kpss(name)
        self._compute_kurtosis(name)

    def _compute_adfuller(self, name: str):
        # performs the Augmented Dickey-Fuller unit root test for stationarity
        adf_stat, p_value_adf, _, _, critical_vals_adf, _ = adfuller(self[name], regression='ct')
        stats = {
            'ADF Statistic': adf_stat,
            'p-value': p_value_adf,
            'Critical values': critical_vals_adf
        }
        self.__stats[name]['Dickey-Fuller'] = stats

    def _compute_kpss(self, name: str):
        # performs the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
        kpss_stat, p_value_kpss, _, critical_vals_kpss = kpss(self[name], regression='ct', nlags='auto')

        stats = {
            'KPSS Statistic': kpss_stat,
            'p-value': p_value_kpss,
            'Critical values': critical_vals_kpss
        }
        self.__stats[name]['KPSS'] = stats

    def _compute_kurtosis(self, name: str):
        # calculates the kurtosis
        stats = {
            'Kurtosis Statistic': self[name].kurtosis()
        }
        self.__stats[name]['Kurtosis'] = stats


if __name__ == '__main__':
    t = TimeSeriesAnalysis()
    t.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv", index_col='Month')
    t.stats('Red ')
