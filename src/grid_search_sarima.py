import itertools
import os

from src.TimeSeries.TimeSeries import TimeSeries
from src.Utils.Utils import Utils

if __name__ == "__main__":
    repo_name = 'wine_market_temporal_prediction'
    local_path = os.getcwd().split(repo_name)[0] + repo_name
    data_path = os.path.join(local_path, 'data/AustralianWines.csv')

    t = TimeSeries()
    t.load(file_path=data_path, index_col='Month')

    col_names = t.col_names()
    wine_sel = col_names[5]
    time_series = t[wine_sel]

    txt_path = os.path.join(local_path, f'data/parameters_tried_{wine_sel.strip()}.txt')

    print('Columns names: \n', *col_names)
    print('Grid Search for wine type: ', wine_sel)

    p = P = q = Q = d = D = [0, 1, 2]
    m = [12]
    pdqPDQm = list(itertools.product(p, d, q, P, D, Q, m))

    Utils.fit_SARIMA_txt(txt_path, time_series, pdqPDQm, maxiter=50)
    best_params, best_aic = Utils.read_params_aic(txt_path, best=True)

    print(f'Best params for wine type {wine_sel}: \n', f'{best_params}  --> AIC : {best_aic}')
