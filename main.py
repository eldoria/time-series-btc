import pandas as pd

from models import return_prediction
from evaluate_models import evaluate_model
from vizualize_data import print_data_and_prediction

data = pd.read_csv("data/clean_historic/BTCUSDT-1m-2024-03.csv")
data["date"] = pd.to_datetime(data["date"])

nb_test_lines = 1000
train_data = data[:(len(data)-nb_test_lines)]
test_data = data.tail(nb_test_lines).reset_index()

prediction_data = return_prediction(train_data, "naive_model", nb_test_lines).reset_index()
loss = evaluate_model(test_data.price, prediction_data.price, "mse")
print_data_and_prediction(test_data, prediction_data)

