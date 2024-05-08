import pandas as pd
import os
from datetime import datetime


# with windows size of 100 il will the 501 last prices values
def add_lag(df: pd.DataFrame, lags=[1, 2, 3, 10, 50, 100, 200]) -> pd.DataFrame:
    for lag in lags:
        df[f"price_shifted_{lag}"] = df["price"].shift(periods=lag, fill_value=-1)
    return df


# bitcoin has a 4 years seasonality until Hyperbitconization
def add_cycle_information(df: pd.DataFrame) -> pd.DataFrame:
    for cycle_year in [0, 1, 2, 3]:
        df[f"cycle_year_{cycle_year}"] = df["year"].apply(lambda x: 1 if (x - 2014) % 4 == cycle_year else 0)
    return df


# halving cuts inflation of bitcoin in half every four years approximately, they generate euphoria in the market
def add_halving_dates(df: pd.DataFrame):
    halving_dates = [datetime.strptime(halving_date, "%Y-%m-%d") \
                     for halving_date in ["2016-07-09", "2020-05-11", "2024-04-20"]]
    df["date"] = df.apply(lambda x: f"{int(x['year'])}-{int(x['month'])}-{int(x['day'])}", axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["last_halving"] = df["date"].apply(
        lambda x: min([(x - halving_date).days for halving_date in halving_dates if (x - halving_date).days >= 0]))
    df["next_halving"] = df["date"].apply(
        lambda x: max([(x - halving_date).days for halving_date in halving_dates if (x - halving_date).days <= 0]))
    return df.drop("date", axis=1)


# add the features to the dataset and save it
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_halving_dates(
        add_cycle_information(
            add_lag(
                df
            )
        )
    )


# only 1 file because the lag features make other aggregation than overall data not relevant
file = "BTCUSDT-1d.csv"
path_to_data_folder = "/".join(os.path.abspath(__file__).split("/")[:-1])
data_path = f"{path_to_data_folder}/clean_historic/{file}"
new_path = f"{path_to_data_folder}/clean_historic_with_features/{file}"

df = pd.read_csv(data_path)
df_with_features = add_features(df)

df_with_features.to_csv(new_path, index=False)
