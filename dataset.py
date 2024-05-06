import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path_df, train_test_ratio):
        self.df = pd.read_csv(path_df)
        self.features = self.df.drop("price", axis=1).columns

        self.train_test_ratio = train_test_ratio
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def prepare_data_lstm(self, window_size):
        """Use that when doing predictions with window historic (LSTM)"""
        X, y = [], []
        for i in range(len(self.df) - window_size):
            X.append(self.df.iloc[i:(i + window_size)][self.features].values)
            y.append(self.df.iloc[i + window_size]['price'])

        # add axis for batch
        X = np.array(X)
        y = np.array(y)

        # Split the data into training and testing sets
        train_size = int(len(self.df) * self.train_test_ratio)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def prepare_data_xgboost(self, window_size):
        X = self.df.drop("price", axis=1)
        y = self.df["price"]

        train_size = int(len(self.df) * self.train_test_ratio)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def extend_data_with_n_steps(self, df, n_steps):
        """Extend given data with n steps using predictions as data for lag prices"""

        df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
        for i in range(n_steps):
            new_date = df["date"].iloc[-1] + timedelta(1)
            new_day = new_date.day
            new_month = new_date.month
            new_year = new_date.year

            new_price_shifted_1 = df.iloc[-1].price_shifted_1
            new_price_shifted_2 = df.iloc[-2].price_shifted_2
            new_price_shifted_3 = df.iloc[-3].price_shifted_3
            new_price_shifted_10 = df.iloc[-10].price_shifted_10
            new_price_shifted_50 = df.iloc[-50].price_shifted_50
            new_price_shifted_100 = df.iloc[-100].price_shifted_100
            new_price_shifted_200 = df.iloc[-200].price_shifted_200

            new_cycle_year_0 = 1 if (new_year - 2014) % 4 == 0 else 0
            new_cycle_year_1 = 1 if (new_year - 2014) % 4 == 1 else 0
            new_cycle_year_2 = 1 if (new_year - 2014) % 4 == 2 else 0
            new_cycle_year_3 = 1 if (new_year - 2014) % 4 == 3 else 0

            new_last_halving = df["last_halving"].iloc[-1] + 1
            new_next_halving = df["next_halving"].iloc[-1] + 1

            if new_next_halving == 0:
                new_last_halving = 0
            if new_next_halving == 1:
                new_next_halving = -365 * 4

            new_row = {
                "year": new_year,
                "month": new_month,
                "day": new_day,

                "price_shifted_1": new_price_shifted_1,
                "price_shifted_2": new_price_shifted_2,
                "price_shifted_3": new_price_shifted_3,
                "price_shifted_10": new_price_shifted_10,
                "price_shifted_50": new_price_shifted_50,
                "price_shifted_100": new_price_shifted_100,
                "price_shifted_200": new_price_shifted_200,

                "cycle_year_0": new_cycle_year_0,
                "cycle_year_1": new_cycle_year_1,
                "cycle_year_2": new_cycle_year_2,
                "cycle_year_3": new_cycle_year_3,

                "last_halving": new_last_halving,
                "next_halving": new_next_halving
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        return df.drop("date", axis=1)
