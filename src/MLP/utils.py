import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from random import random
from datetime import datetime as dt
from pandas import DatetimeIndex
from src.MLP.mlp_models import WineDataset


def model_eval(trained_model: nn.Module, dataset: WineDataset):
    """
    Takes a WineDataset and returns a pandas series (using datetime indexing) with the predictions.
    """
    trained_model.eval()
    y_pred = []
    with torch.no_grad():
        for index in range(len(dataset)):
            temp_seq, true_pred, temp_seq_index, true_pred_index = dataset.get_with_index(index)
            outputs = trained_model(temp_seq)
            y_pred.append(pd.Series(outputs.numpy(), index=true_pred_index))
    return pd.concat(y_pred)

