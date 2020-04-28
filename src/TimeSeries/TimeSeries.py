import pandas as pd
from typing import List
import matplotlib.pyplot as plt


class TimeSeries:
    """
    Class to handle generic temporal series. The series must be stored in a csv file where each row is a temporal
    record and each column a different temporal serie.
    """

    def __init__(self):
        self._data = None
        self._col_names = None
        self._index_name = None

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
        self._data.set_index(pd.to_datetime(self._data[self._index_name]), inplace=True)
        self._data.index.freq = self._data.index.inferred_freq
        for col_name in self._col_names:
            self._data[col_name] = pd.to_numeric(self._data[col_name], errors='coerce')
            self._data[col_name].interpolate(method='time', inplace=True)

    @staticmethod
    def _get_customized_figure():
        """
        This method should work as a style template for all the plots.
        """
        fig, ax = plt.subplots()
        return fig, ax

    def plot_serie(self, name: str, ax: plt.Axes, start: str):
        """
        Plots a temporal serie.

        :param name:                    Name of the serie that wants to be plotted.
        :param ax:                      Matplotlib axis where data will be plotted.
        :param start:                         String representation of a date from which predict.
        """
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
    t.plot_serie('Red ')
