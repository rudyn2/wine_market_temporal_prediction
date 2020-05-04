from src.TimeSeries.TimeSeriesMA import TimeSeriesMA

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    name = 'Rose '

    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    train, val = TimeSeriesMA(), TimeSeriesMA()
    train.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTrain.csv", index_col='Month')
    val.load("/Users/rudy/Documents/wine_market_temporal_prediction/data/AustralianWinesTest.csv", index_col='Month')
    train.plot_serie(name, ax=axs[0])
    val.plot_serie(name, ax=axs[1])

    train.difference()
    train.fit_scale()

    val.difference()
    val = train.scale(val)

    train.inv_scale()
    train.inv_difference()

    val.inv_scale()
    val.inv_difference()

    train.plot_serie(name, ax=axs[0])
    axs[0].set(xlabel='Fecha', ylabel='Miles de litros', title='Entrenamiento')

    # plot results
    val.plot_serie(name, ax=axs[1])
    axs[1].set(xlabel='Fecha', ylabel='Miles de litros', title='Validación')

    plt.legend()
    plt.show()
