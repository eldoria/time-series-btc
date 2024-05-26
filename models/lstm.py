from models import Models
from dataset import Dataset

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout


import numpy.ma as ma


class Lstm(Models):
    def __init__(self, path_df, train_test_ratio, window_size):
        super().__init__()

        self.dataset = Dataset(path_df, train_test_ratio)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.prepare_data(window_size)

        self.features = self.dataset.columns

        self.window_size = window_size
        self.online_data = pd.DataFrame()

        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(window_size, len(self.features))),
            Dropout(0.5),
            Dense(1)
        ])

    def train(self, nb_epochs, nb_batch_size):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print(ma.shape(self.X_train))
        print(ma.shape(self.y_train))
        print(self.X_train)
        print(self.y_train)
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=128, verbose=1)

        train_loss = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f'Train Loss: {train_loss: .4f}')
        print(f'Test Loss: {test_loss: .4f}')

    '''
    def online_learning(self, online_data):
        index = online_data["index"]
        print(index)
        if index == 0:
            return

        brute_data = [int(data) for data in online_data["data"].split(',')]

        data = pd.DataFrame(columns=self.features+["price"])
        data.loc[len(data)] = brute_data
        self.online_data = pd.concat([self.online_data, data])

        if index <= self.window_size+1:
            return
        else:
            self.online_data = self.online_data.iloc[1:, :]

        self.create_x_y(self.online_data)

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.X, self.y, epochs=1, batch_size=1, verbose=1)
    '''
    def predict(self, data, n_steps):
        data_to_predict = self.dataset.extend_data_with_n_steps(data, n_steps)

        predictions = self.model.predict(data_to_predict)

        return np.array(predictions)

    def save_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        save_model(self.model, path_to_file)

    def load_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        return load_model(path_to_file)


if __name__ == "__main__":
    lstm = Lstm(
        path_df="../data/clean_historic/BTCUSDT-1d.csv",
        train_test_ratio=0.5,
        window_size=10
    )
    lstm.train(
        nb_epochs=300,
        nb_batch_size=128
    )
