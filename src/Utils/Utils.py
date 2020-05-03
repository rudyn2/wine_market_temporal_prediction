import itertools
import os
from typing import List

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


