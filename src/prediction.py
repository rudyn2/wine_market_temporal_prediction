#!/usr/bin/env python

"""
prediction.py: This scripts is a short how-to-guide of statsmodel.api used in a case study of
SARIMAX model. This code is not extensible and it just must be used as an example.
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
import itertools
import statsmodels.api as sm

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

__author__ = "Rudy García A."

# importing files
raw_data = pd.read_csv("../data/AustralianWines.csv")

# pre-processing
raw_data['Month'] = raw_data['Month'].apply(lambda x: dt.strptime(str(x), '%b-%y'))
raw_data.set_index(raw_data['Month'], inplace=True)
wine_names = [column_name for column_name in raw_data.columns if column_name != "Month"]
for wine_name in wine_names:
    raw_data[wine_name] = pd.to_numeric(raw_data[wine_name], errors='coerce')
    raw_data[wine_name].interpolate(method='time', inplace=True)

# plotting decomposition
wine_name = 'Rose '
t_serie = raw_data[wine_name]
decomposition = sm.tsa.seasonal_decompose(t_serie, model='additive')

plt.figure(figsize=(10, 16))
decomposition.plot()
plt.show()

# generates list of SARIMAX order's
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# fitting
mod = sm.tsa.statespace.SARIMAX(t_serie,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

# results
print(results.summary())
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# making the prediction
pred = results.get_prediction(start=pd.to_datetime('1994-10-01'), dynamic=False)
pred_ci = pred.conf_int()

# plots the prediction and the bounds
ax = t_serie['1993-01-01':].plot(label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel(f'{wine_name} Wine Sales')
plt.legend()
plt.show()

