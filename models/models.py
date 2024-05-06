from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools


class ModelLSTM:
    def __init__(self, features, window_size=100):
        self.window_size = window_size
        self.features = features

        self.model = Sequential([
            LSTM(10, activation='relu', input_shape=(window_size, len(features))),
            # Dropout(0.5),
            Dense(1)
        ])
        self.online_data = pd.DataFrame()
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def create_x_y(self, training_data):
        # scaler = MinMaxScaler()

        # prices_2d = training_data['price'].values.reshape(-1, 1)

        # training_data["price_scaled"] = scaler.fit_transform(prices_2d)

        X, y = [], []
        for i in range(len(training_data) - self.window_size):
            X.append(training_data.iloc[i:(i + self.window_size)][self.features].values)
            y.append(training_data.iloc[i + self.window_size]['price'])
        self.X = np.array(X)
        self.y = np.array(y)

        return X, y

    def train(self, training_data):
        self.create_x_y(training_data)

        # Split the data into training and testing sets
        train_size = int(len(self.X) * 0.8)
        X_train, X_test = self.X[:train_size], self.X[train_size:]
        y_train, y_test = self.y[:train_size], self.y[train_size:]

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(X_train, y_train, epochs=200, batch_size=1024, verbose=1)

        # Evaluate the model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Train Loss: {train_loss: .4f}')
        print(f'Test Loss: {test_loss: .4f}')

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

    def predict(self, input_sequence, nb_steps):
        predictions = []
        current_input = input_sequence
        for _ in range(nb_steps):
            # Predict the next step ahead
            prediction = self.model.predict(self.X_test)
            # Append the prediction to the list
            if not predictions:
                predictions = prediction
            else:
                predictions.append(prediction[-1])

        return np.array(predictions)

    def save_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        save_model(self.model, path_to_file)

    def load_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        return load_model(path_to_file)


df = pd.read_csv("../data/clean_historic_with_features/BTCUSDT-1d.csv")
lstm = ModelLSTM(features=df.drop("price", axis=1).columns, window_size=100)
# lstm.model = lstm.load_model()

if __name__ == "__main__":
    lstm.train(df)
    lstm.create_x_y(df)

    y = lstm.y_test.reshape(-1)
    y = pd.Series(y, name="price")

    y_predicted = lstm.predict(lstm.X_test, 1)
    y_predicted = y_predicted.reshape(-1)
    y_predicted = pd.Series(y_predicted, name="price")

    print(y_predicted)
    y.plot(kind="line")
    y_predicted.plot(kind="line")
    plt.show()

    lstm.save_model()

