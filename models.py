# return the last value of the time series
import pandas as pd


def naive_model(data):
    return data["price"].iloc[-1]


def return_prediction(data, model, steps):
    time_interval = data["date"].iloc[1] - data["date"].iloc[0]
    for _ in range(steps):
        last_time_interval = data["date"].iloc[-1]
        if model == "naive_model":
            prediction_price = naive_model(data)
            new_row = {"date": last_time_interval + time_interval, "price": prediction_price}
            # remove first row and add new row
            data = data[1:]
            data = pd.concat([data, pd.DataFrame([new_row])])
    return data.tail(steps)
