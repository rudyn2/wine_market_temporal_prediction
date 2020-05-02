#!/usr/bin/env python

from datetime import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# importing files
raw_data = pd.read_csv("../data/AustralianWines.csv")

# pre-processing
raw_data['Month'] = raw_data['Month'].apply(lambda x: dt.strptime(str(x), '%b-%y'))
wine_names = [column_name for column_name in raw_data.columns if column_name != "Month"]
for wine_name in wine_names:
    raw_data[wine_name] = pd.to_numeric(raw_data[wine_name], errors='coerce')

# renaming to spanish
new_names = {
    'Fortified': 'Fortificado',
    'Red ': 'Tinto',
    'Rose ': 'Rosa',
    'sparkling ': 'Espumoso',
    'Sweet white': 'Blanco dulce',
    'Dry white': 'Blanco seco'
}

raw_data.rename(columns=new_names, inplace=True)
wine_names = list(new_names.values())

# first visualization
fig, axs = plt.subplots(3, 2, dpi=300, figsize=(16, 12))
for index, ax in enumerate(axs.reshape(-1)):
    ax.plot(raw_data['Month'], raw_data[wine_names[index]])
    ax.set(xlabel='Fecha', ylabel='Miles de litros', title=wine_names[index])
plt.show()
