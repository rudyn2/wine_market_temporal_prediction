from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from collections import defaultdict


class DiffOperation:

    def __init__(self):
        self._data_copy: pd.Series = pd.Series(dtype='float64')
        self._interval: int = 0

    def fit_transform(self, data: pd.Series, interval: int = 1) -> pd.Series:
        self._data_copy = data.copy(deep=True)
        self._interval = interval
        diff = []
        indexes = []
        for index in range(interval, len(data)):
            value = data.iloc[index] - data.iloc[index - interval]
            indexes.append(data.index[index])
            diff.append(value)
        return pd.Series(data=diff, index=indexes).dropna()

    def invert(self, external_diff: pd.Series) -> pd.Series:

        # assert len(external) == len(self._data_copy) - self._interval
        assert type(external_diff.index) is pd.DatetimeIndex

        inverted_indexes = [self._data_copy.index[i] for i in range(self._interval)]
        inverted_values = [self._data_copy.iloc[i] for i in range(self._interval)]

        for index in external_diff.index:
            # datetime index
            int_loc = list(self._data_copy.index).index(index)
            value = external_diff.loc[index] + self._data_copy.iloc[int_loc - self._interval]
            inverted_values.append(value)
            inverted_indexes.append(index)
        inverter_ts = pd.Series(data=np.float64(inverted_values), index=inverted_indexes)
        return inverter_ts

    def partial_invert(self, external: pd.Series) -> pd.Series:

        assert type(external.index) is pd.DatetimeIndex

        inverted_indexes = []
        inverted_values = []

        for index in external.index:
            # datetime index
            int_loc = list(self._data_copy.index).index(index)
            value = external.loc[index] + self._data_copy.iloc[int_loc - self._interval]
            inverted_values.append(value)
            inverted_indexes.append(index)

        return pd.Series(data=inverted_values, index=inverted_indexes).copy()


class TimeSeries:
    """
    Class to handle generic temporal series. The series must be stored in a csv file where each row is a temporal
    record and each column a different temporal serie.
    """

    def __init__(self):
        self._diff_operators: defaultdict[list] = defaultdict(list)
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._is_scaled = False
        self._data: pd.DataFrame = pd.DataFrame()
        self._col_names: list = []
        self._index_name: str = ''

    def load(self, file_path: str, index_col: str):
        """
        Loads a csv with temporal series.

        :param file_path:                       Path of file to the data.
        :param index_col:                       Name of the index column.
        """
        self._data = pd.read_csv(file_path)
        self._col_names = [column_name for column_name in self._data.columns if column_name != index_col]
        self._index_name = index_col
        self._preprocess()

    def _preprocess(self):
        """
        Does general pre-processing of the temporal series.
        """
        if self._index_name is not None:
            self._data.set_index(pd.to_datetime(self._data[self._index_name]), inplace=True)
        self._data.index.freq = self._data.index.inferred_freq
        for col_name in self._col_names:
            self._data[col_name] = pd.to_numeric(self._data[col_name], errors='coerce')
            self._data[col_name].interpolate(method='time', inplace=True)
        self._data.drop(columns=[self._index_name], inplace=True)

    def timeseries_to_supervised(self, name: str, lag: int = 1, width: int = 4, pred_width: int = 2) -> Tuple[
        np.array, np.array, list, list]:
        """
        Transform the data from temporal series to (x, y) supervised data. It returns two numpy arrays with data.

        :param name:                            Name of particular serie that wants to be transformed.
        :param lag:                             Number of periods of lag.
        :param width:                           Width of the temporal window of data.
        :param pred_width:                      Width of prediction window.
        :return:                                A Tuple.
                                                    1) numpy array with X elements.
                                                    2) numpy array with y elements.
                                                    3) temporal indexes of X elements.
                                                    4) temporal indexes of y elements.
        """
        series = self._data[name].copy()
        dataset_x = []
        dataset_y = []
        dataset_x_index = []
        dataset_y_index = []

        for index in range(len(series) - width - lag - pred_width + 1):
            x_element = series.iloc[index:index + width]
            y_element = series.iloc[index + width + lag:index + width + lag + pred_width]
            dataset_x.append(np.array(x_element))
            dataset_y.append(np.array(y_element))
            dataset_x_index.append(series.index[index:index + width])
            dataset_y_index.append(series.index[index + width + lag:index + width + lag + pred_width])

        return np.array(dataset_x), np.array(dataset_y), dataset_x_index, dataset_y_index

    def difference(self, interval: int = 1):
        """
        Differences the data for an interval.

        :param interval:                        Delta difference.
        :return:                                A copy of the data after the operation.
        """

        assert type(interval) is int, "Just integer values for interval parameter are allowed."
        for name in self._col_names:
            diff_op = DiffOperation()
            self._data[name] = diff_op.fit_transform(self._data[name], interval)
            self._diff_operators[name].append(diff_op)
        self._data.dropna(inplace=True)
        return self.copy()

    def inv_difference(self):
        """
        Reverse the last difference operation.
        """
        new_data = {}
        for name in self._col_names:
            new_data[name] = self._diff_operators[name].pop().invert(self._data[name])
        self._data = pd.DataFrame(new_data)
        return self.copy()

    def inv_difference_serie(self, name: str, external_serie: pd.Series) -> pd.Series:
        """
        Reverse the difference of external data using difference values stored in the last difference
        operation made by this object.
        """
        if name not in self._diff_operators.keys():
            raise ValueError("Invalid operation")
        return self._diff_operators[name].partial_invert(external_serie)

    def scale(self):
        """
        Scales the data using a MinMaxScaler.
        """
        self._data[:] = self._scaler.fit_transform(X=self._data)
        self._is_scaled = True
        return self.copy()

    def inv_scale(self):
        """
        Reverse the last scale operation.
        """
        self._data[:] = self._scaler.inverse_transform(X=self._data)
        self._is_scaled = False
        return self.copy()

    def inv_scale_serie(self, name: str, external_serie: pd.Series) -> pd.Series:
        """
        Reverse the scale of data using Min Max values stored in the last scaling operation made by this object.

        :param name:                                Name of the reference serie.
        :param external_serie:                      Serie that wants to be reversed.
        """
        if self._is_scaled:
            loc = self._col_names.index(name)
            col_min = self._scaler.min_[loc]
            col_scale = self._scaler.scale_[loc]
            external_serie -= col_min
            external_serie /= col_scale
            return external_serie
        return external_serie

    def copy(self):
        """
        Makes a copy of a TimeSeries.
        """
        new = TimeSeries()
        new._data = self._data.copy()
        new._index_name = self._index_name
        new._col_names = self._col_names
        new._index_name = self._index_name
        new._scaler = self._scaler
        new._is_scaled = self._is_scaled
        new._diff_operators = self._diff_operators
        return new

    def plot_with(self, name: str, external_serie: pd.Series):
        """
        Makes a plot of the common values between the external serie and the column called 'name'.
        """
        intersection = self._data[name].loc[external_serie.index]
        if len(intersection) == 0:
            raise ValueError(f"There are not common values between col '{name}' and "
                             "provided external series.")
        fig, ax = self._get_customized_figure()
        internal: pd.Series = self[name].loc[intersection.index]
        external = external_serie.loc[intersection.index]
        internal.plot(ax=ax, label='internal')
        external.plot(ax=ax, label='external')
        ax.legend()
        plt.show()

    @staticmethod
    def _get_customized_figure():
        """
        This method should work as a style template for all the plots.
        """
        fig, ax = plt.subplots()
        return fig, ax

    def plot_serie(self, name: str, ax: plt.Axes, start: str = ''):
        """
        Plots a temporal serie.

        :param name:                    Name of the serie that wants to be plotted.
        :param ax:                      Matplotlib axis where data will be plotted.
        :param start:                         String representation of a date from which predict.
        """
        if start == '':
            start = self._data.index[0]
        self._data[name][start:].plot(ax=ax, label='Observaciones')
        ax.legend()

    def col_names(self) -> List[str]:
        return self._col_names

    def info(self):
        return self._data.info()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = TimeSeries()
    t.load(file_path='/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv',
           index_col='Month')
    name = 'Red '
    fig, ax = plt.subplots()
    t[name].plot(ax=ax)
    t.scale()

    # we extract a part of the serie and sums 1 to each value
    subt_copy = t[name].copy()
    subt_copy = subt_copy[10:40]

    subt_inv_scaled = t.inv_scale_serie(name, subt_copy)
    subt_inv_scaled.plot(ax=ax, label='external inv scaled')
    t[name].plot(ax=ax, label='original scaled')
    plt.legend()
    plt.show()
