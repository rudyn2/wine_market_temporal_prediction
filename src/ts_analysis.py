#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns

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

"""
ts_analysis.py: This scripts executes several analysis on available data. It shows the most meaningful statistics
associated to the nature of the time series being analyzed.
"""

from src.TimeSeries.TimeSeriesAnalysis import TimeSeriesAnalysis

if __name__ == '__main__':
    data_path = '/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWines.csv'
    t = TimeSeriesAnalysis()
    t.load(data_path, index_col='Month')
    wine_names = t.col_names()

    # plots histograms
    fig, axs = plt.subplots(2, 3, dpi=300, figsize=(18, 12))
    for index, ax in enumerate(axs.reshape(-1)):
        t.plot_hist(wine_names[index], bins=20, ax=ax)
        ax.set(xlabel='Miles de litros', title=wine_names[index])
    plt.show()
