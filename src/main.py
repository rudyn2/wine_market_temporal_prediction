"""
main.py:

Shows the results for sarimax, MA and MLP models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.MLP.mlp_models import MLP, WineDataset
from src.MLP.utils import model_eval
from src.TimeSeries.TimeSeries import TimeSeries
from src.TimeSeries.TimeSeriesForecast import TimeSeriesForecast

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

if __name__ == '__main__':
    random_seed = 42
    input_size = 12
    output_size = 1
    test_size = 1 / 8
    name = 'Rose '
    mlp_model_path = f'/Users/rudy/Documents/wine_market_temporal_prediction/data/model_{name}.pt'
    mlp = True
    sarimax = True
    ma = False

    # for auxiliary purposes
    train_ts = TimeSeries()
    train_ts.load('/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTrain.csv',
                  index_col='Month')
    val_ts = TimeSeries()
    val_ts.load('/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTest.csv',
                index_col='Month')

    # region: MLP
    if mlp:
        t = TimeSeries()
        t.load('/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv', index_col='Month')
        t.difference()
        t.scale()
        x, y, x_index, y_index = t.timeseries_to_supervised(name=name, width=input_size, pred_width=output_size)

        # split
        X_train, X_valid, y_train, y_valid, x_index_train, x_valid_index, y_train_index, y_index_val = \
            train_test_split(x, y, x_index, y_index, test_size=test_size, random_state=random_seed, shuffle=False)

        train_dataset = WineDataset(x=torch.from_numpy(X_train).float(), y=torch.from_numpy(y_train).float(),
                                    x_index=x_index_train, y_index=y_train_index)
        valid_dataset = WineDataset(x=torch.from_numpy(X_valid).float(), y=torch.from_numpy(y_valid).float(),
                                    x_index=x_valid_index, y_index=y_index_val)

        model: nn.Module = MLP(input_size, output_size)
        model.load_state_dict(torch.load(mlp_model_path))

        train_mlp_ts = model_eval(model, dataset=train_dataset)
        val_mlp_ts = model_eval(model, dataset=valid_dataset)

        # using t properties to reverse the operations
        train_mlp_ts = t.inv_scale_serie(name=name, external_serie=train_mlp_ts)
        train_mlp_ts = t.inv_difference_serie(name=name, external_serie=train_mlp_ts)
        val_mlp_ts = t.inv_scale_serie(name=name, external_serie=val_mlp_ts)
        val_mlp_ts = t.inv_difference_serie(name=name, external_serie=val_mlp_ts)

        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        # plot results
        train_mlp_ts.plot(ax=axs[0], label='Predicción')
        train_ts.plot_serie(name, ax=axs[0])
        axs[0].set(xlabel='Fecha', ylabel='Miles de litros', title='Entrenamiento')

        val_mlp_ts.plot(ax=axs[1], label='Predicción')
        val_ts.plot_serie(name, ax=axs[1])
        axs[1].set(xlabel='Fecha', ylabel='Miles de litros', title='Validación')

        plt.show()

    # endregion: MLP

    # region: SARIMAX
    if sarimax:
        sarimax_train_ts = TimeSeriesForecast()
        sarimax_train_ts.load('/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTrain.csv',
                              index_col='Month')
        order = (1, 0, 0)
        seasonal_order = (1, 0, 0, 12)
        sarimax_train_ts.fit_sarimax(name, order, seasonal_order)
        train_sarimax_pred, train_sarimax_pred_ci = sarimax_train_ts.predict_in_sample(name)
        val_sarimax_pred, val_sarimax_pred_ci = sarimax_train_ts.predict_out_of_sample(name,
                                                                                       start=val_ts[name].index[0],
                                                                                       end=val_ts[name].index[-1])
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        sarimax_train_ts.plot_serie(name, ax=axs[0])
        train_sarimax_pred.plot(ax=axs[0], label='Predicción')
        axs[0].set(xlabel='Fecha', ylabel='Miles de litros', title='Entrenamiento')

        val_ts.plot_serie(name, ax=axs[1])
        val_sarimax_pred.plot(ax=axs[1], label='Predicción')
        axs[1].set(xlabel='Fecha', ylabel='Miles de litros', title='Validación')

        plt.show()
    # endregion
