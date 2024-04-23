import pandas as pd
import os


brute_paths = ["data/brute_historic/" + file for file in os.listdir("data/brute_historic") if ".csv" in file]
clean_paths = ["data/clean_historic/" + file for file in os.listdir("data/clean_historic") if ".csv" in file]


columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_time",
           "number_of_trades", "taker_buy_base_asset_volume", "ignore"]

for brute_path in brute_paths:
    brute_file = brute_path.split("/")[-1]
    if brute_file not in [clean_path.split("/")[-1] for clean_path in clean_paths]:
        df = pd.read_csv(brute_path, usecols=[0, 4], header=None)
        df.columns = ["timestamp", "price"]
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["hour"] = df["date"].dt.hour
        df["minute"] = df["date"].dt.minute
        df[["timestamp", "date", "year", "month", "day", "hour", "minute", "price"]]\
            .to_csv(f"data/clean_historic/" + brute_file, index=False)
        print(f"data/clean_historic/{brute_file} created")

clean_paths = ["data/clean_historic/" + file for file in os.listdir("data/clean_historic") if ".csv" in file]
years_prefix = list(set(["-".join(name_file.split("-")[:-1]) for name_file in clean_paths if "20" in "-".join(name_file.split("-")[:-1])]))

for year_prefix in years_prefix:
    globals()[year_prefix] = pd.DataFrame()

for clean_path in clean_paths:
    year_prefix = "-".join(clean_path.split("-")[:-1])
    if '20' not in year_prefix:
        continue
    globals()[year_prefix] = pd.concat([globals()[year_prefix], pd.read_csv(clean_path)])

all_data = pd.DataFrame()
for year_prefix in years_prefix:
    globals()[year_prefix].sort_values(by=['timestamp'], inplace=True)
    globals()[year_prefix].to_csv(year_prefix+'.csv', index=False)
    print(f"{year_prefix+'.csv'} created")
    time_interval = year_prefix.split('-')[1]
    all_data = pd.concat([all_data, globals()[year_prefix]])

all_data.sort_values(by=['timestamp'], inplace=True)
all_data.to_csv(f"data/clean_historic/BTCUSDT-1d.csv", index=False)
print(f"BTCUSDT-1d.csv created")
