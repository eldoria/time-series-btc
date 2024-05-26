import pandas as pd
import matplotlib.pyplot as plt

from models.xgb_regressor import XgbRegressor


xgb_regressor = XgbRegressor(
    path_df="data/clean_historic_with_features/BTCUSDT-1d.csv",
    train_test_ratio=0.8,
    window_size=10
)
'''
xgb_regressor.train(
    nb_epochs=300,
    nb_batch_size=128
)
xgb_regressor.save_model()
'''
xgb_regressor.load_model()

y_train = pd.Series(xgb_regressor.y_train)
y_test = pd.Series(xgb_regressor.y_test)

df_train = xgb_regressor.X_train
df_train["price"] = y_train

y_predicted = xgb_regressor.predict(df_train, len(xgb_regressor.X_test))
y_predicted = pd.Series(y_predicted)

'''
#y_predicted.index = y_test.index

y_train.plot(kind="line", style="b-")
y_test.plot(kind="line", style="b:")
y_predicted.plot(kind="line", style="r--")
plt.show()
'''
