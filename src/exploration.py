import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime as dt
sns.set()

# importing files
raw_data = pd.read_csv("../data/AustralianWines.csv")

# pre-processing
raw_data['Month'] = raw_data['Month'].apply(lambda x: dt.strptime(str(x), '%b-%y'))
wine_names = [column_name for column_name in raw_data.columns if column_name != "Month"]
for wine_name in wine_names:
    raw_data[wine_name] = pd.to_numeric(raw_data[wine_name], errors='coerce')

# first visualization
fig, axs = plt.subplots(3, 2, dpi=300, figsize=(16, 12))
for index, ax in enumerate(axs.reshape(-1)):
    ax.plot(raw_data['Month'], raw_data[wine_names[index]])
    ax.set(xlabel='Date', ylabel='Trade units', title=wine_names[index])
plt.show()
