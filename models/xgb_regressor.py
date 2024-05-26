from models.models import Models
from dataset import Dataset

import os
import pandas as pd
import xgboost as xgb


class XgbRegressor(Models):
    def __init__(self, path_df="data/clean_historic_with_features/BTCUSDT-1d.csv", train_test_ratio=0.8, window_size=1):
        super().__init__()

        path_df = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]), path_df)

        self.online_data = pd.DataFrame()  # for real time prediction
        self.dataset = Dataset(path_df, train_test_ratio)  # for batch prediction
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.prepare_data_xgboost(window_size)

        self.columns = self.dataset.columns
        self.features = self.dataset.features

        self.window_size = window_size

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            booster='gbtree',
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.5  # each tree only use 80% of the training, i use it for breaking the determinism
        )

    def train(self, nb_epochs, nb_batch_size):
        self.model.fit(self.X_train, self.y_train,
                       eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)], early_stopping_rounds=20)

    def online_learning(self, online_data):
        index = online_data["index"]

        if index == 0:  # header
            return

        brute_data = [int(data) for data in online_data["data"].split(',')]

        data = pd.DataFrame(columns=self.columns)

        data.loc[len(data)] = brute_data
        self.online_data = pd.concat([self.online_data, data])

        X, y = data[self.features], data["price"]

        self.model.fit(X, y, verbose=1)

    def predict(self, df, n_steps):
        for _ in range(n_steps):
            # print(df.dtypes)
            df = self.dataset.extend_data_with_1_step(df)
            # print(df.dtypes)
            X_for_prediction = pd.DataFrame(df[self.features].iloc[-1]).transpose()
            prediction = int(self.model.predict(X_for_prediction)[0])
            df.loc[len(df)-1, "price"] = prediction

        df.to_csv("test.csv", index=False)
        return df["price"].iloc[len(df)-n_steps:]

    def save_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        self.model.save_model(path_to_file)

    def load_model(self):
        path_to_file = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-1]), "lstm.keras")
        self.model.load_model(path_to_file)
