#!/usr/bin/env python

"""
grid_search_sarima.py: this script is used for grid-search of optimal parameters for a SARIMA model of prediction
                      of a TimeSeries. It reads from a file saved in data/ folder and saves parameters already tried
                      alongside their AIC metric, for comparative motives. Finally, prints the best parameters among
                      tried.
"""

import itertools
import os

from src.TimeSeries.TimeSeries import TimeSeries
from src.Utils.Utils import Utils

if __name__ == "__main__":
    repo_name = Utils.get_repo_path()

    ts = TimeSeries()
    ts.load(os.path.join(repo_name, 'data/AustralianWines.csv'), index_col='Month')

    col_names = ts.col_names()
    wine_sel = col_names[5]
    time_series = ts[wine_sel]

    txt_path = os.path.join(repo_name, f'data/parameters_tried_{wine_sel.strip()}.txt')

    print('Columns names: \n', *col_names)
    print('Grid Search for wine type: ', wine_sel)

    p = P = q = Q = d = D = [0, 1, 2]
    m = [12]
    pdqPDQm = list(itertools.product(p, d, q, P, D, Q, m))

    Utils.fit_SARIMA_txt(txt_path, time_series, pdqPDQm, maxiter=50)
    best_params, best_aic = Utils.read_params_aic(txt_path, best=True)

    print(f'Best params for wine type {wine_sel}: \n', f'{best_params}  --> AIC : {best_aic}')
