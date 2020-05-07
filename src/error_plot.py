#!/usr/bin/env python

"""
error_plot.py: a script for plotting absolute error histograms for every wine type in data, also displays the
              confidence interval at 80% for each case, using 10% and 90% percentile approximation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from src.MLP.mlp_models import MLP, WineDataset
from src.MLP.utils import model_eval
from src.TimeSeries.TimeSeriesSarimax import TimeSeriesSarimax
from src.Utils.Loss import TimeSeriesErrorMetric as TSEM
from src.Utils.Utils import Utils

sns.set()
Utils.set_plot_config()


if __name__ == '__main__':
    repo_path = Utils.get_repo_path()
    np.random.seed(42)
    input_size = 12
    output_size = 1
    test_size = 1 / 8
    mlp = True
    sarimax = False

    train_ts, val_ts, entire_ts = Utils.train_test_data(os.path.join(repo_path, 'data/AustralianWines.csv'))
    names = entire_ts.col_names()
    # region: MLP
    if mlp:
        fig, axs = plt.subplots(nrows=len(names) // 2, ncols=2, figsize=(10, 16), dpi=180)

        for i, name in enumerate(names):
            mlp_model_path = os.path.join(repo_path, f'data/model_{name}.pt')
            t_train, t_valid, _ = Utils.train_test_data(os.path.join(repo_path, 'data/AustralianWines.csv'))

            t_train.difference()
            t_train.fit_scale()
            X_train, y_train, x_train_index, y_train_index = t_train.timeseries_to_supervised(name=name,
                                                                                              width=input_size,
                                                                                              pred_width=output_size)

            t_valid.difference()
            X_valid, y_valid, x_val_index, y_val_index = t_train.scale(t_valid).timeseries_to_supervised(name=name,
                                                                                                         width=input_size,
                                                                                                         pred_width=output_size)

            train_dataset = WineDataset(x=torch.from_numpy(X_train).float(), y=torch.from_numpy(y_train).float(),
                                        x_index=x_train_index, y_index=y_train_index)
            valid_dataset = WineDataset(x=torch.from_numpy(X_valid).float(), y=torch.from_numpy(y_valid).float(),
                                        x_index=x_val_index, y_index=y_val_index)

            model: nn.Module = MLP(input_size, output_size)
            model.load_state_dict(torch.load(mlp_model_path))

            val_mlp_ts = model_eval(model, dataset=valid_dataset)

            # using t properties to reverse the operations
            val_mlp_ts = t_valid.inv_scale_serie(name=name, external_serie=val_mlp_ts)
            val_mlp_ts = t_valid.inv_difference_serie(name=name, external_serie=val_mlp_ts)

            error, per10, per90 = TSEM.abs_error_w_conf(val_ts[name], val_mlp_ts)
            axs[i // 2, i % 2].hist(error)
            axs[i // 2, i % 2].set(title=f'Error para {name}', xlabel='Error absoluto', ylabel='Frecuencia')

            print(f'C.I. 80% para {name}:', f'({per10}, {per90})')
        plt.show()

    # region: SARIMAX
    if sarimax:
        fig, axs = plt.subplots(nrows=len(names) // 2, ncols=2, figsize=(10, 16), dpi=180)

        sarima_train_ts = TimeSeriesSarimax()
        sarima_train_ts.load(os.path.join(repo_path, 'data/AustralianWinesTrain.csv'), index_col='Month')
        for i, name in enumerate(names):
            sarima_train_ts.load_params_file(name, os.path.join(repo_path, f'data/parameters_tried_{name.strip()}.txt'))
            sarima_train_ts.fit_on_best(name)
            val_sarimax_pred, _ = sarima_train_ts.predict_out_of_sample(name,
                                                                        start=val_ts[name].index[0],
                                                                        end=val_ts[name].index[-1])

            error, per10, per90 = TSEM.abs_error_w_conf(val_ts[name], val_sarimax_pred)
            axs[i // 2, i % 2].hist(error)
            axs[i // 2, i % 2].set(title=f'Error para {name}', xlabel='Error absoluto', ylabel='Frecuencia')

            print(f'C.I. 80% para {name}:', f'({per10}, {per90})')
        plt.show()
