import os
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tools.eval_measures import mse

from src.TimeSeries.TimeSeries import TimeSeries
from src.Utils.Utils import Utils


class TimeSeriesErrorMetric:
    """
    Class that handles the error metrics for TimeSeries prediction results.
    Includes both MSE and MAPE evaluation metrics.
    """

    @staticmethod
    def __get_series_intersection(serie_x: pd.Series, serie_y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        It returns the points of the series where they have the same index.

        :param serie_x:         a pd.Series object
        :param serie_y:         a pd.Series object
        :return:                a tuple of series with same index
                                        1) pd.Series
                                        2) pd.Series
        """
        series_intersection_index = serie_x.index.intersection(serie_y.index)
        s1 = serie_x[series_intersection_index]
        s2 = serie_y[series_intersection_index]
        return s1, s2

    @classmethod
    def MSETimeSeries(cls, serie_x: pd.Series, serie_y: pd.Series) -> float:
        """
        Computes the Mean Square Error metric of two TimeSeries, computed element-wise.

        :param serie_x:         a pd.Series object
        :param serie_y:         a pd.Series object
        :return:                a float number that represents the MSE metric
        """
        s1, s2 = cls.__get_series_intersection(serie_x, serie_y)
        return mse(s1, s2)

    @classmethod
    def MAPETimeSeries(cls, serie_x: pd.Series, serie_y: pd.Series) -> float:
        """
        Computes the Mean Absolute Percentage Error metric of two TimeSeries, computed element-wise.

        :param serie_x:         a pd.Series object
        :param serie_y:         a pd.Series object
        :return:                a float number that represents the MAPE metric
        """
        s1, s2 = cls.__get_series_intersection(serie_x, serie_y)
        mape = np.sum(np.divide(100 * np.abs(s1 - s2), s1)) / len(s1)
        return mape

    @classmethod
    def abs_error_w_conf(cls, serie_x: pd.Series, serie_y: pd.Series, conf_perc: float = 0.8) -> \
            Tuple[np.array, float, float]:
        """
        It returns the absolute error of two series element-wise in an numpy.array and also returns an approximation
        of the upper and lower limits of a confidence interval of *conf_perc* proportion. The background for this is
        that an interval of c*100% can be approximated by the percentiles (alpha, 1-alpha) where alpha can be obtained
        as alpha = (1 - c) / 2.

        :param serie_x:         a pd.Series object
        :param serie_y:         a pd.Series object
        :param conf_perc:       confidence proportion of the confidence interval
        :return:                a tuple of
                                            1) a np.array with error element-wise
                                            2) a float percentile 100*alpha
                                            3) a float percentile (1-alpha)*100
        """
        s1, s2 = cls.__get_series_intersection(serie_x, serie_y)

        alpha = (1 - conf_perc) / 2
        error = np.abs(s1 - s2)
        per_alpha = np.percentile(error, int(100*alpha))
        per_alphac = np.percentile(error, int(100*(1-alpha)))

        return error, per_alpha, per_alphac


if __name__ == "__main__":
    repo_path = Utils.get_repo_path()
    series = TimeSeries()
    series.load(os.path.join(repo_path, 'data/AustralianWines.csv'), index_col='Month')

    names = series.col_names()
    s_1 = series[names[0]]
    s_2 = series[names[1]]

    mse_metric = TimeSeriesErrorMetric.MSETimeSeries(s_1, s_2)
    mape_metric = TimeSeriesErrorMetric.MAPETimeSeries(s_1, s_2)

    print(f'MSE between {names[0]} and {names[1]} : {mse_metric:.4f}')
    print(f'MAPE between {names[0]} and {names[1]} : {mape_metric:.4f}%')
