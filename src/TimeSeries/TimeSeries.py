import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler


class TimeSeries:
    """
    Class to handle generic temporal series. The series must be stored in a csv file where each row is a temporal
    record and each column a different temporal serie.
    """

    def __init__(self):
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._data: pd.DataFrame = None
        self._org_data: pd.DataFrame = None
        self._col_names: list = []
        self._index_name: str = ''
        self._diff_interval: int = 0
        self._diff_init_values: pd.DataFrame = None

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

    def timeseries_to_supervised(self, name, lag=1, width=4, pred_width=2) -> Tuple[List[pd.Series], List[float]]:
        """
        Transform the data from temporal series to (x, y) supervised data. It returns two numpy arrays with data.

        :param name:                            Name of particular serie that wants to be transformed.
        :param lag:                             Number of periods of lag.
        :param width:                           Width of the temporal window of data.
        :param pred_width:                      Width of prediction window.
        :return:                                A Tuple.
                                                    1) X elements.
                                                    2) Y elements.
        """
        series = self._data[name].copy()
        dataset_x = []
        dataset_y = []

        for index in range(len(series)-width-lag-pred_width+1):
            x_element = series.iloc[index:index+width]
            y_element = series.iloc[index+width+lag:index+width+lag+pred_width]
            dataset_x.append(x_element)
            dataset_y.append(y_element)

        return dataset_x, dataset_y

    def difference(self, interval: int = 1):
        """
        Differences the data for an interval.

        :param interval:                        Delta difference.
        :return:                                A copy of the data after the operation.
        """

        assert type(interval) is int, "Just integer values for interval parameter are allowed."
        self._diff_init_values = self._data.iloc[:interval]
        self._data = self._data.diff(interval)
        self._diff_interval = interval
        return self.copy()

    def inv_difference(self):
        """
        Reverse the last difference operation.
        """
        self._data.fillna(value=self._diff_init_values, inplace=True)
        for i in range(self._diff_interval, len(self._data)):
            self._data.iloc[i, :] = self._data.iloc[i, :] + self._data.iloc[i-self._diff_interval, :]
        return self.copy()

    def scale(self):
        """
        Scales the data using a MinMaxScaler.
        """
        self._data[:] = self._scaler.fit_transform(X=self._data)
        return self.copy()

    def inv_scale(self):
        """
        Reverse the last scale operation.
        """
        self._data[:] = self._scaler.inverse_transform(X=self._data)
        return self.copy()

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
        new._diff_interval = self._diff_interval
        new._diff_init_values = self._diff_init_values
        return new

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
        self._data[name][start:].plot(ax=ax, label='Observations')
        ax.legend()

    def col_names(self) -> List[str]:
        return self._col_names

    def info(self) -> str:
        return self._data.info()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


if __name__ == '__main__':
    t = TimeSeries()
    t.load(file_path='/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv',
           index_col='Month')
    t.info()
    x, y = t.timeseries_to_supervised('Red ', lag=0, width=1, pred_width=1)
    t.scale()
    t.inv_scale()
    t.difference()
    t.inv_difference()