"""
main.py:

Shows the results for sarimax, MA and MLP models.
"""

import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from statsmodels.tools.eval_measures import mse

from src.MLP.mlp_models import MLP, WineDataset
from src.MLP.utils import model_eval
from src.TimeSeries.TimeSeries import TimeSeries
from src.TimeSeries.TimeSeriesSarimax import TimeSeriesSarimax
from src.Utils.Utils import Utils

sns.set()

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def histogram_error(serie_x: pd.Series, serie_y: pd.Series):
    series_intersection_index = serie_x.index.intersection(serie_y.index)
    s1 = serie_x[series_intersection_index]
    s2 = serie_y[series_intersection_index]

    error = np.abs(s1-s2)
    per_10 = np.percentile(error, 10)
    per_90 = np.percentile(error, 90)

    return error, per_10, per_90


if __name__ == '__main__':
    repo_path = Utils.get_repo_path()
    np.random.seed(42)
    input_size = 12
    output_size = 1
    test_size = 1 / 8
    name = 'Fortified'
    mlp_model_path = os.path.join(repo_path, f'data/model_{name}.pt')
    mlp = False
    sarimax = True
    ma = False

    train_ts, val_ts, entire_ts = TimeSeries(), TimeSeries(), TimeSeries()
    train_ts.load(os.path.join(repo_path, 'data/AustralianWinesTrain.csv'), index_col='Month')
    val_ts.load(os.path.join(repo_path, 'data/AustralianWinesTest.csv'), index_col='Month')
    entire_ts.load(os.path.join(repo_path, 'data/AustralianWines.csv'), index_col='Month')

    # region: SARIMAX
    if sarimax:
        names = entire_ts.col_names()
        fig, axs = plt.subplots(nrows=len(names) // 2, ncols=2, figsize=(10, 16), dpi=180)
        _i = 0

        for i, name in enumerate(names):
            sarimax_train_ts = TimeSeriesSarimax()
            sarimax_train_ts.load(os.path.join(repo_path, 'data/AustralianWinesTrain.csv'), index_col='Month')

            txt_path = os.path.join(repo_path, f'data/parameters_tried_{name.strip()}.txt')
            best_params, _ = Utils.read_params_aic(txt_path, best=True)

            order = best_params[:3]
            seasonal_order = best_params[3:]
            sarimax_train_ts.fit(name, order=order, seasonal_order=seasonal_order)
            train_sarimax_pred, train_sarimax_pred_ci = sarimax_train_ts.predict_in_sample(name)
            val_sarimax_pred, val_sarimax_pred_ci = sarimax_train_ts.predict_out_of_sample(name,
                                                                                           start=val_ts[name].index[0],
                                                                                           end=val_ts[name].index[-1])

            error, per10, per90 = histogram_error(val_ts[name], val_sarimax_pred)
            axs[i // 2, _i].hist(error)
            axs[i // 2, _i].set(title=f'Error para {name}', xlabel='Error absoluto', ylabel='Frecuencia')

            _i = (_i + 1) % 2

            print(f'C.I. 80% para {name}:', f'({per10}, {per90})')

        plt.show()
