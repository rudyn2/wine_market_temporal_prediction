import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.MLP.mlp_models import WineDataset, MLP
from torch.utils.data import DataLoader

from src.TimeSeries.TimeSeries import TimeSeries

sns.set()

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def train(model, optimizer, loss_fn, train_loader, valid_loader, epochs: int = 10):
    mean_train_losses = []
    mean_valid_losses = []

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        # train step
        for i, (input_seq, output_seq) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(input_seq)
            loss = loss_fn(outputs, output_seq)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # calculates validation loss
        model.eval()
        with torch.no_grad():
            for i, (input_seq, output_seq) in enumerate(valid_loader):
                outputs = model(input_seq)
                loss = loss_fn(outputs, output_seq)
                valid_losses.append(loss.item())

        # saves losses
        mean_train_losses.append(np.mean(train_losses))
        mean_valid_losses.append(np.mean(valid_losses))

        # prints progress
        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}' \
              .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses)))
    return mean_train_losses, mean_valid_losses


if __name__ == '__main__':
    i_shape = 12
    o_shape = 1
    name = 'Rose '

    t = TimeSeries()
    t.load('/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv', index_col='Month')
    t.difference()
    t.scale()
    x, y, x_index, y_index = t.timeseries_to_supervised(name=name, width=i_shape, pred_width=o_shape)

    # split
    X_train, X_valid, y_train, y_valid, x_train_index, x_valid_index, y_train_index, y_val_index = \
        train_test_split(x, y, x_index, y_index, test_size=1 / 8, random_state=42, shuffle=False)

    X_train = torch.from_numpy(X_train).float()  # convert to tensors
    y_train = torch.from_numpy(y_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    y_valid = torch.from_numpy(y_valid).float()

    train_dataset = WineDataset(X_train, y_train, x_train_index, y_train_index)
    valid_dataset = WineDataset(X_valid, y_valid, x_valid_index, y_val_index)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=128, shuffle=False)

    model: nn.Module = MLP(i_shape, o_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    train_loss, valid_loss = train(model, optimizer, loss_fn, train_loader, valid_loader, epochs=100)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set(xlabel='Epochs', ylabel='MSE Loss', title="MSE Loss evolution vs Epochs")
    ax.plot(train_loss, label='train')
    ax.plot(valid_loss, label='valid')
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')
    plt.show()

    torch.save(model.state_dict(), f'/Users/rudy/Documents/wine_market_temporal_prediction/data/model_{name}.pt')
