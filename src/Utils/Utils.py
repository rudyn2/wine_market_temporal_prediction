import itertools
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from src.TimeSeries.TimeSeries import TimeSeries


class Utils:
    """
    Class that handles tasks that are related to TimeSeries objects but are not meant to be
    included into the same class as the object.
    Utils contains static methods as it's not to be seen as and object itself but as set of
    complementary functionality to TimeSeries objects.
    """

    @staticmethod
    def fit_SARIMA(data: pd.Series, params: tuple, maxiter: int = 50, disp_summary: str = False):
        """
        Fits a SARIMA model for the given data and params where the first must be a time series and the second must
        come in the form (p,d,q,P,D,Q,m). The number of iterations can be adjusted with maxiter. A summary of the
        results can be displayed with disp_summary.

        :param data:                    a Pandas Series that represents the Time Series
        :param params:                  regular and seasonal params of the SARIMA model
        :param maxiter:                 maximum number of iterations for fitting the model
        :param disp_summary:            a flag to activate the display of results' summary

        :return:                        the result of fitting the model and the model itself -> (result, model)
        """
        assert len(params) == 7, f"not enough params given, expected 7, given {len(params)}"
        model = sm.tsa.statespace.SARIMAX(data,
                                          order=params[:3],
                                          seasonal_order=params[3:],
                                          enforce_stationarity=False,
                                          enforce_invertibility=False
                                          )
        results = model.fit(maxiter=maxiter, disp=False)

        if disp_summary:
            results.summary()

        return results, model

    @classmethod
    def fit_SARIMA_txt(cls, txt_file: str, data: pd.Series, params: List[tuple], **kwargs):
        """
        Fits a model SARIMA for every set of parameters passed in params, but first checking that these values
        haven't already been fitted in which case they should be in txt_file. Every new sets of params gets written
        into txt_file alongside its AIC metric, in order to keep the file updated.

        :param txt_file:            file that contains params already tried for fitting and their AIC metric
        :param data:                a Pandas Series that represents the Time Series
        :param params:              regular and seasonal params of the SARIMA model
        :param kwargs:              extra params that can be passed to fit_SARIMA method in Utils class
        """
        old_params, _ = cls.read_params_aic(txt_file, best=False)
        new_params = list(set(params) - set(old_params))

        with open(txt_file, 'a') as f:
            for params in new_params:
                try:
                    _result, _model = cls.fit_SARIMA(data, params, **kwargs)
                    print(f'SARIMAX: {params[:3]}x{params[3:6]}{params[6]} - AIC:{_result.aic}')
                    f.write(f"{params} && AIC: {_result.aic}\n")
                except Exception:
                    continue

    @staticmethod
    def read_params_aic(txt_path: str, best: bool = True):
        """
        Get the list of params for SARIMA model that have already been fitted, with their associated
        AIC metric, with the possibility of returning only the parameters with best AIC (smallest).

        :param txt_path:            file that contains params already tried for fitting and their AIC metric
        :param best:                if true then returns only the best parameters and AIC, else return all
        :return:                    a List of tuples and a List of floats
        """
        params_tried = []
        aic_metric = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    params_tried += [eval(line.replace('\n', '').split("&&")[0])]
                    aic_metric += [float(line.replace('\n', '').split(":")[-1])]
                if best:
                    idxmax = min(range(len(aic_metric)), key=lambda i: aic_metric[i])
                    return params_tried[idxmax], aic_metric[idxmax]
        except FileNotFoundError:
            print("Parameters tried file not found")

        return params_tried, aic_metric

    @staticmethod
    def get_repo_path():
        """
        :return: local repository path independent of pc running program
        """
        repo_name = 'wine_market_temporal_prediction'
        local_path = os.getcwd().split(repo_name)[0] + repo_name

        return local_path

    @staticmethod
    def train_test_data(file_path_b: str, idx_col: str = 'Month'):
        """
        It loads the train and test data of a time series alongside the full time series. It assumes that the files
        are in the same folder specified by *file_path_b* and the names of the file are computed as:
            train_path: file_path_b - '.csv' + 'Train.csv'
            test_path:  file_path_b - '.csv' + 'Test.csv'
        Then it also specifies the index column *idx_col*.

        :param file_path_b:     path to the full TimeSeries CSV file.
        :param idx_col:         index column of the data
        :return:                a tuple of
                                            1) train data TimeSeries object
                                            2) test data TimeSeries object
                                            3) full data TimeSeries object
        """
        base_path = file_path_b.replace('.csv', '')
        train_path = base_path + 'Train.csv'
        test_path = base_path + 'Test.csv'

        train_ts, val_ts, full_ts = TimeSeries(), TimeSeries(), TimeSeries()
        train_ts.load(train_path, index_col=idx_col)
        val_ts.load(test_path, index_col=idx_col)
        full_ts.load(file_path_b, index_col=idx_col)

        return train_ts, val_ts, full_ts

    @staticmethod
    def set_plot_config(s: int = 16, m: int = 16, b: int = 22):
        """
        Set global sizes for plots in three possible sizes given by small *s*, medium *m* and big *b*
        """
        plt.rc('font', size=s)          # controls default text sizes
        plt.rc('axes', titlesize=s)     # fontsize of the axes title
        plt.rc('axes', labelsize=m)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=s)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=s)    # fontsize of the tick labels
        plt.rc('legend', fontsize=s)    # legend fontsize
        plt.rc('figure', titlesize=b)   # fontsize of the figure title


if __name__ == "__main__":
    repo_path = Utils.get_repo_path()
    txt_path = os.path.join(repo_path, 'data/parameters_tried.txt')
    data_path = os.path.join(repo_path, 'data/AustralianWines.csv')
    t = TimeSeries()
    t.load(file_path=data_path, index_col='Month')

    col_names = t.col_names()
    time_series = t[col_names[0]]

    p = P = q = Q = d = D = [0]
    m = [12]
    pdqPDQm = list(itertools.product(p, d, q, P, D, Q, m))

    Utils.fit_SARIMA_txt(txt_path, time_series, pdqPDQm, maxiter=25)

    best_params, best_aic = Utils.read_params_aic(txt_path, best=True)
