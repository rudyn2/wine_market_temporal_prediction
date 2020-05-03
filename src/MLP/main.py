import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from src.Utils.Utils import Utils

from src.MLP.mlp_models import MLP, WineDataset
from src.MLP.utils import model_eval
from src.TimeSeries.TimeSeriesAnalysis import TimeSeriesAnalysis
import matplotlib.pyplot as plt
from src.TimeSeries.TimeSeries import TimeSeries

if __name__ == '__main__':
    repo_path = Utils.get_repo_path()
    input_size = 12
    output_size = 1
    name = 'Fortified'
    model_path = f'/Users/rudy/Documents/wine_market_temporal_prediction/data/model_{name}.pt'

    t_train, t_valid = TimeSeries(), TimeSeries()
    t_train.load(os.path.join(repo_path, 'data/AustralianWinesTrain.csv'), index_col='Month')
    t_valid.load(os.path.join(repo_path, 'data/AustralianWinesTest.csv'), index_col='Month')

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
    model.load_state_dict(torch.load(model_path))
    model.eval()

    valid_pred = model_eval(model, dataset=valid_dataset)

    # using t properties to reverse the operations
    valid_pred = t_valid.inv_scale_serie(name=name, external_serie=valid_pred)
    valid_pred = t_valid.inv_difference_serie(name=name, external_serie=valid_pred)

    t_valid.inv_scale()
    t_valid.inv_difference()

    fig, ax = plt.subplots()
    t_valid[name].plot(ax=ax)
    valid_pred.plot(ax=ax)

    plt.show()
