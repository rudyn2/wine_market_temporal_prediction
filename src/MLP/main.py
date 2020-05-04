import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from src.Utils.Utils import Utils
from torch.utils.data import DataLoader
from src.MLP.mlp_models import MLP, WineDataset
from src.MLP.utils import model_eval
from src.TimeSeries.TimeSeriesAnalysis import TimeSeriesAnalysis
import matplotlib.pyplot as plt
from src.TimeSeries.TimeSeries import TimeSeries
from src.MLP.train import train
from src.main import mape

if __name__ == '__main__':
    repo_path = Utils.get_repo_path()
    torch.manual_seed(42)
    input_size = 12
    output_size = 1

    wine_names = ['Fortified', 'Red ', 'Rose ', 'sparkling ', 'Sweet white', 'Dry white']
    fig, axs = plt.subplots(nrows=len(wine_names), ncols=2, figsize=(15, 24), dpi=180, sharey='row')
    losses = {}

    for i, name in enumerate(wine_names):

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

        train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)

        # saving results
        # model: nn.Module = MLP(input_size, output_size)
        # model.load_state_dict(torch.load(model_path))
        # model.eval()

        # train
        model: nn.Module = MLP(input_size, output_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = nn.MSELoss()
        train_loss, valid_loss = train(model, optimizer, loss_fn, train_loader, valid_loader, epochs=70, verbose=False)
        losses[name] = dict(train=train_loss, valid=valid_loss)

        # evaluation
        train_result = model_eval(model, train_dataset)
        val_result = model_eval(model, valid_dataset)

        # inverting prediction
        train_result_inverted = t_train.inv_scale_serie(name, train_result)
        train_result_inverted = t_train.inv_difference_serie(name, train_result_inverted)
        val_result_inverted = t_valid.inv_scale_serie(name, val_result)
        val_result_inverted = t_valid.inv_difference_serie(name, val_result_inverted)

        # inverting originals
        t_valid.inv_scale()
        t_valid.inv_difference()
        t_train.inv_scale()
        t_train.inv_difference()

        print(f"{name} train mape: {mape(train_result_inverted, t_train[name])} | "
              f"valid mape: {mape(val_result_inverted, t_valid[name])}")

        torch.save(model.state_dict(), os.path.join(repo_path, f'data/model_{name}.pt'))

        # plot train
        train_result_inverted.plot(ax=axs[i, 0], label='Observaciones')
        t_train[name].plot(ax=axs[i, 0], label='Predicción', title=name)
        axs[i, 0].legend()

        # plot valid
        val_result_inverted.plot(ax=axs[i, 1], label='Observaciones')
        t_valid[name].plot(ax=axs[i, 1], label='Predicción', title=name)
        axs[i, 1].legend()

    axs[0, 0].set(title=f'Entrenamiento\n{wine_names[0]}')
    axs[0, 1].set(title=f'Validación\n{wine_names[0]}')

    for ax in axs.flat:
        ax.set(xlabel='Fecha', ylabel='Miles de litros')
    for ax in axs.flat:
        ax.label_outer()

    plt.show()

    # plots learning curves
    for name in losses.keys():
        loss = losses[name]
        train_loss = loss['train']
        valid_loss = loss['valid']

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set(xlabel='Epochs', ylabel='MSE Loss', title=f"MSE Loss evolution vs Epochs ({name})")
        ax.plot(train_loss, label='train')
        ax.plot(valid_loss, label='valid')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')
        plt.show()
