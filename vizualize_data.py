import matplotlib.pyplot as plt

import pandas as pd


def print_data(data):
    plt.plot(data.index, data.price)
    plt.show()


def print_data_and_prediction(observed_data, prediction_data):
    if type(observed_data) is pd.DataFrame:
        plt.plot(observed_data.index, observed_data.price)
    else:
        plt.plot(observed_data.index, observed_data)
    '''
    if type(observed_data) is pd.DataFrame:
        plt.plot(prediction_data.index, prediction_data.price)
    else:
        plt.plot(prediction_data.index, prediction_data)
    '''
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/clean_historic/BTCUSDT-1m-2024-03.csv")
    print_data(df)
