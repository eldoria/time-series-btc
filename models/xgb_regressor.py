from models.models import Models
from dataset import Dataset

import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split


class XgbRegressor(Models):
    def __init__(self, path_df, train_test_ratio, window_size):
        super().__init__()

        self.dataset = Dataset(path_df, train_test_ratio)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.prepare_data_xgboost(window_size)

        self.features = self.dataset.features

        self.window_size = window_size
        self.online_data = pd.DataFrame()

        self.model = xgb.XGBRegressor(
            objective='reg:linear',
            booster='gbtree',
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.5  # each tree only use 80% of the training, i use it for breaking the determinism
        )

    def train(self, nb_epochs, nb_batch_size):
        self.model.fit(self.X_train, self.y_train,
                       eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds=20)

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
        self.model.save_model(path_to_file)

    def load_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        self.model.load_model(path_to_file)
