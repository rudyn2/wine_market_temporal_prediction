#!/usr/bin/env python

"""
main.py:

Shows the results for sarimax, MA and MLP models.
"""

import os
from pprint import pprint

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
    ma = False

    train_ts, val_ts, total_ts = Utils.train_test_data(os.path.join(repo_path, 'data/AustralianWines.csv'))
    names = total_ts.col_names()
    # region: MLP
    if mlp:
        fig, axs = plt.subplots(nrows=len(names), ncols=2, figsize=(15, 24), dpi=180, sharey='row')
        MSE_reg = {'train': {}, 'val': {}}

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

            model_mlp: nn.Module = MLP(input_size, output_size)
            model_mlp.load_state_dict(torch.load(mlp_model_path))

            train_mlp_ts = model_eval(model_mlp, dataset=train_dataset)
            val_mlp_ts = model_eval(model_mlp, dataset=valid_dataset)

            # using t properties to reverse the operations
            train_mlp_ts = t_train.inv_scale_serie(name=name, external_serie=train_mlp_ts)
            train_mlp_ts = t_train.inv_difference_serie(name=name, external_serie=train_mlp_ts).dropna()

            # using t properties to reverse the operations
            val_mlp_ts = t_valid.inv_scale_serie(name=name, external_serie=val_mlp_ts)
            val_mlp_ts = t_valid.inv_difference_serie(name=name, external_serie=val_mlp_ts)

            # using t properties to reverse the operations
            val_mlp_ts = t_valid.inv_scale_serie(name=name, external_serie=val_mlp_ts)
            val_mlp_ts = t_valid.inv_difference_serie(name=name, external_serie=val_mlp_ts)

            # plot results
            train_mlp_ts.plot(ax=axs[i, 0], label='Predicción', title=name)
            train_ts.plot_serie(name, ax=axs[i, 0])
            axs[i, 0].legend()

            # plot results
            val_mlp_ts.plot(ax=axs[i, 1], label='Predicción')
            val_ts.plot_serie(name, ax=axs[i, 1])
            axs[i, 1].legend()

            MSE_reg['train'][name] = TSEM.MSETimeSeries(train_ts[name], train_mlp_ts)
            MSE_reg['val'][name] = TSEM.MSETimeSeries(val_ts[name], val_mlp_ts)

        axs[0, 0].set(title=f'Entrenamiento\n{names[0]}')
        axs[0, 1].set(title=f'Validación\n{names[0]}')

        for ax in axs.flat:
            ax.set(xlabel='Fecha', ylabel='Miles de litros')
        for ax in axs.flat:
            ax.label_outer()
        plt.suptitle('Resultados con MLP')
        plt.show()

        print("MLP Metrics")
        pprint(MSE_reg, width=1)
    # endregion: MLP

    # region: SARIMA
    if sarimax:
        names = total_ts.col_names()
        fig, axs = plt.subplots(nrows=len(names), ncols=2, figsize=(15, 24), dpi=180, sharey='row')
        MAPE_reg = {'train': {}, 'val': {}}

        sarima_train_ts = TimeSeriesSarimax()
        sarima_train_ts.load(os.path.join(repo_path, 'data/AustralianWinesTrain.csv'), index_col='Month')
        for i, name in enumerate(names):
            sarima_train_ts.load_params_file(name, os.path.join(repo_path, f'data/parameters_tried_{name.strip()}.txt'))
            sarima_train_ts.fit_on_best(name)
            train_sarima_pred, _ = sarima_train_ts.predict_in_sample(name)
            val_sarima_pred, _ = sarima_train_ts.predict_out_of_sample(name,
                                                                       start=val_ts[name].index[0],
                                                                       end=val_ts[name].index[-1])
            MAPE_reg['train'][name] = TSEM.MAPETimeSeries(train_ts[name], train_sarima_pred)
            MAPE_reg['val'][name] = TSEM.MAPETimeSeries(val_ts[name], val_sarima_pred)

            sarima_train_ts.plot_serie(name, ax=axs[i, 0])
            train_sarima_pred.plot(ax=axs[i, 0], label='Predicción', title=name)
            axs[i, 0].legend()

            val_ts.plot_serie(name, ax=axs[i, 1])
            val_sarima_pred.plot(ax=axs[i, 1], label='Predicción', title=name)
            axs[i, 1].legend()

        axs[0, 0].set(title=f'Entrenamiento\n{names[0]}')
        axs[0, 1].set(title=f'Validación\n{names[0]}')

        for ax in axs.flat:
            ax.set(xlabel='Fecha', ylabel='Miles de litros')
        for ax in axs.flat:
            ax.label_outer()
        plt.suptitle('Resultados con SARIMA')
        plt.show()

        print('SARIMA Metrics')
        pprint(MAPE_reg, width=1)
    # endregion

    # region: MA
    if ma:
        name = 'Fortified'

        ma_train_ts = TimeSeriesSarimax()
        ma_train_ts.load(os.path.join(repo_path, 'data/AustralianWinesTrain.csv'), index_col='Month')
        ma_train_ts.fit(name, order=(0, 2, 12), seasonal_order=(0, 0, 0, 0))

        train_ma_pred, _ = ma_train_ts.predict_in_sample(name)
        val_ma_pred, _ = ma_train_ts.predict_out_of_sample(name,
                                                           start=val_ts[name].index[0],
                                                           end=val_ts[name].index[-1])

        fig, axs = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(16, 8))

        train_ts[name].plot(ax=axs[0], label='true')
        train_ma_pred.plot(ax=axs[0], label='predicted')
        axs[0].set(xlabel='Fecha', ylabel='Miles de litros', title='Entrenamiento')

        val_ts[name].plot(ax=axs[1], label='true')
        val_ma_pred.plot(ax=axs[1], label='forecast')
        axs[1].set(xlabel='Fecha', ylabel='Miles de litros', title='Validación')

        plt.legend()
        plt.suptitle('Resultados con MA y 2-diff')
        plt.show()
    # endregion
