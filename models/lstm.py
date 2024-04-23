from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import numpy as np
import matplotlib.pyplot as plt

from vizualize_data import print_data_and_prediction


class ModelLSTM:
    def __init__(self, features, window_size=1000):
        self.window_size = window_size
        self.features = features

        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(window_size, len(features))),
            Dropout(0.5),
            Dense(1)
        ])

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data.iloc[i:(i + self.window_size)][self.features].values)
            y.append(data.iloc[i + self.window_size]['price_scaled'])
        return np.array(X), np.array(y)

    def return_data(self, training_data):
        scaler = MinMaxScaler()

        prices_2d = training_data['price'].values.reshape(-1, 1)

        training_data["price_scaled"] = scaler.fit_transform(prices_2d)

        return self.create_sequences(training_data)

    def train(self, training_data):
        X, y = self.return_data(training_data)

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(X_train, y_train, epochs=300, batch_size=128, verbose=1)

        # Evaluate the model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')

    def predict(self, input_sequence, nb_steps):
        predictions = []
        current_input = input_sequence
        for _ in range(nb_steps):
            # Predict the next step ahead
            prediction = self.model.predict(current_input)
            # Append the prediction to the list
            predictions.append(prediction)
            # Update the input sequence for the next step
            # current_input = np.concatenate([current_input[1:], prediction], axis=0)
        return np.array(predictions)

    def save_model(self):
        save_model(self.model, "lstm.keras")

    def load_model(self):
        load_model("lstm.keras")


if __name__ == "__main__":
    import pandas as pd

    lstm = ModelLSTM(features=["year", "month", "day", "hour", "minute"], window_size=20)
    df = pd.read_csv("../data/clean_historic/BTCUSDT-1d.csv")

    lstm.train(df)
    X, y = lstm.return_data(df)

    y = y.reshape(-1)
    y = pd.Series(y, name="price")

    y_predicted = lstm.predict(X, 1)
    y_predicted = y_predicted.reshape(-1)
    y_predicted = pd.Series(y_predicted, name="price")

    y.plot(kind="line")
    y_predicted.plot(kind="line")
    plt.show()

    lstm.save_model()

