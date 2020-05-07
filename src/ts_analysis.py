#!/usr/bin/env python

"""
ts_analysis.py: This scripts executes several analysis on available data. It shows the most meaningful statistics
               associated to the nature of the time series being analyzed.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.TimeSeries.TimeSeriesAnalysis import TimeSeriesAnalysis
from src.Utils.Utils import Utils

sns.set()
Utils.set_plot_config()


if __name__ == '__main__':
    repo_path = Utils.get_repo_path()
    ts = TimeSeriesAnalysis()
    ts.load(os.path.join(repo_path, 'data/AustralianWines.csv'), index_col='Month')
    wine_names = ts.col_names()

    fig, axs = plt.subplots(2, 3, dpi=300, figsize=(18, 12))
    for index, ax in enumerate(axs.reshape(-1)):
        ts.plot_hist(wine_names[index], bins=20, ax=ax)
        ax.set(xlabel='Miles de litros', title=wine_names[index])
    plt.show()
