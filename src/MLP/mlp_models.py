import torch.nn as nn
from torch.utils.data import Dataset


class WineDataset(Dataset):
    def __init__(self, x, y, x_index, y_index):
        self.X = x
        self.y = y
        self.x_index = x_index
        self.y_index = y_index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item, :], self.y[item, :]

    def get_with_index(self, item):
        """
        Returns values and its respective datetime indexes.
        """
        return self.X[item, :], self.y[item, :], self.x_index[item], self.y_index[item]


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, output_size)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
