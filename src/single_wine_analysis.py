#!/usr/bin/env python

"""
single_wine_analysis.py: This script is meant to show the visualization of properties of a given time series, in this
case a single wine of all that were provided. It also executes some hypothesis test that give more information on the
stationarity of the series. This code is an initial approach but its use might be extended to other time series in
the future.
"""

from datetime import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
sns.set()

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

__author__ = "Tomás Saldivia A."

# importing files
raw_data = pd.read_csv("../data/AustralianWines.csv")

# pre-processing
raw_data['Month'] = raw_data['Month'].apply(lambda x: dt.strptime(str(x), '%b-%y'))
raw_data.set_index(raw_data['Month'], inplace=True)
wine_names = [column_name for column_name in raw_data.columns if column_name != "Month"]
for wine_name in wine_names:
    raw_data[wine_name] = pd.to_numeric(raw_data[wine_name], errors='coerce')
    raw_data[wine_name].interpolate(method='time', inplace=True)

# selecting a single wine among possible
wine_name = wine_names[0]
data_selected = raw_data[wine_name]

# performs the Augmented Dickey-Fuller unit root test for stationarity
adf_stat, p_value_adf, _, _, critical_vals_adf, _ = adfuller(data_selected, regression='ct')

print('ADF Statistic: %f' % adf_stat)
print('p-value: %f' % p_value_adf)
print('Critical Values:')
for key, value in critical_vals_adf.items():
    print('\t%s: %.3f' % (key, value))

# performs the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
kpss_stat, p_value_kpss, _, critical_vals_kpss = kpss(data_selected, regression='ct', nlags='auto')

print('KPSS Statistic: %f' % kpss_stat)
print('p-value: %f' % p_value_kpss)
print('Critical Values:')
for key, value in critical_vals_kpss.items():
    print('\t%s: %.3f' % (key, value))

# subplot visualization
fig, axs = plt.subplots(2, 2, dpi=200, figsize=(12, 7))

gs = axs[0, 0].get_gridspec()
for ax in axs[0, :]:
    ax.remove()
ax_big = fig.add_subplot(gs[0, :])

# visualizing the time series of the selected wine
data_selected.plot(ax=ax_big)
ax_big.set(xlabel='Date', ylabel='Trade units',
           title=(wine_name+'\n Dickey-Fuller: p={0:.5f} | KPSS: p={1:.5f}'.format(p_value_adf, p_value_kpss)))

# visualizing both the auto-correlation and partial auto-correlation functions
plot_acf(data_selected, lags=int(len(data_selected)/3), ax=axs[1, 0])
plot_pacf(data_selected, lags=int(len(data_selected)/3), ax=axs[1, 1])
plt.tight_layout()
plt.show()
