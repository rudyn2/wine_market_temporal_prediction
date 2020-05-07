#!/usr/bin/env python

"""
visualization.py: a script for visualization of the input data of this work, in specific, the monthly sales
                 of 6 types of australian wines, between 1980 and 1994. The sales are given in
                 thousands of liters.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.TimeSeries.TimeSeries import TimeSeries
from src.Utils.Utils import Utils

sns.set()
Utils.set_plot_config()

if __name__ == '__main__':
    repo_path = Utils.get_repo_path()

    new_names = {
        'Fortified': 'Fortificado',
        'Red ': 'Tinto',
        'Rose ': 'Rosa',
        'sparkling ': 'Espumoso',
        'Sweet white': 'Blanco dulce',
        'Dry white': 'Blanco seco'
    }

    data = TimeSeries()
    data.load(os.path.join(repo_path, 'data/AustralianWines.csv'), index_col='Month')

    names = data.col_names()

    # first visualization
    fig, axs = plt.subplots(3, 2, dpi=300, figsize=(16, 12))
    for i, (name, ax) in enumerate(zip(names, axs.reshape(-1))):
        data[name].plot(ax=ax, title=new_names[name])
    plt.suptitle('Ventas de tipos de vinos', fontsize='xx-large')
    plt.show()
