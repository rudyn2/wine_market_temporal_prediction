import torch
import torch.nn as nn
from src.MLP.mlp_models import MLP, WineDataset
from src.MLP.utils import model_eval
from src.TimeSeries.TimeSeries import TimeSeries
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_path = '/Users/rudy/Documents/wine_market_temporal_prediction/data/model.pt'
    input_size = 12
    output_size = 1
    name='Red '

    model: nn.Module = MLP(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    t = TimeSeries()
    t.load('/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv', index_col='Month')
    t.difference()
    t.scale()
    x, y, x_index, y_index = t.timeseries_to_supervised(name=name, width=input_size, pred_width=output_size)

    # split
    X_train, X_valid, y_train, y_valid, x_index_train, x_valid_index, y_train_index, y_index_val = \
         train_test_split(x, y, x_index, y_index, test_size=1/8, random_state=42, shuffle=False)

    X_train = torch.from_numpy(X_train).float()  # convert to tensors
    y_train = torch.from_numpy(y_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).float()

    train_dataset = WineDataset(x=X_train, y=y_train, x_index=x_index_train, y_index=y_train_index)
    valid_dataset = WineDataset(x=X_valid, y=y_valid, x_index=x_valid_index, y_index=y_index_val)

    a = model_eval(model, dataset=valid_dataset)

    # using t properties to reverse the operations
    b = t.inv_scale_serie(name=name, external_serie=a)
    c = t.inv_difference_serie(name=name, external_serie=b)

    t.inv_scale()
    t.inv_difference()
    t.plot_with(name=name, external_serie=c)
    print("")

    # fig, axs = plt.subplots(nrows=2, ncols=1)
    # axs[0].plot(x_index_train, X_train)
    #
    # y_valid_pred = eval_valid(model, dataset=X_valid)



