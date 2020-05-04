from src.TimeSeries.TimeSeriesMA import TimeSeriesMA

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name = 'Fortified'
    train, val = TimeSeriesMA(), TimeSeriesMA()
    train.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTrain.csv", index_col='Month')
    val.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTest.csv", index_col='Month')

    train.difference(interval=1)
    train.difference(interval=1)

    train.fit(name, order=12)
    val.fit(name, order=12)

    train_ma_pred, train_ma_pred_ci = train.predict_in_sample(name)
    val_ma_pred, val_ma_pred_ci = val.predict_in_sample(name)

    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    train_ma_pred.plot(ax=axs[0], label='Predicción')
    train.plot_serie(name, ax=axs[0])
    axs[0].set(xlabel='Fecha', ylabel='Miles de litros', title='Entrenamiento')

    # plot results
    val_ma_pred.plot(ax=axs[1], label='Predicción')
    val.plot_serie(name, ax=axs[1])
    axs[1].set(xlabel='Fecha', ylabel='Miles de litros', title='Validación')

    plt.legend()
    plt.show()
