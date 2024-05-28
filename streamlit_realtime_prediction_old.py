import pandas as pd
import streamlit as st
import plotly.express as px

import sys
from streamlit import runtime
from streamlit.web import cli as stcli


import os
import subprocess
if runtime.exists():
    pass
else:
    # sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    # sys.exit(stcli.main())
    subprocess.Popen(["streamlit", "run", os.path.abspath(__file__)])


from models.xgb_regressor import XgbRegressor

plot_spot = st.empty()

xgb_regressor = XgbRegressor()
xgb_regressor.load_model()


def update_chart(data, nb_steps=30):
    xgb_regressor.online_learning(data)  # training
    print("function")
    print(xgb_regressor.online_data)

    if xgb_regressor.online_data.empty:
        print("return")
        return

    online_data = xgb_regressor.online_data.reset_index(drop=False)
    print(online_data)

    y_predicted = pd.Series(xgb_regressor.predict(online_data, nb_steps)).reset_index(drop=True)
    df_predicted = pd.DataFrame()
    df_predicted["index"] = pd.Series(
        [index + 2 for index in
         range(online_data["index"].iloc[-1], online_data["index"].iloc[-1] + nb_steps)]
    )
    df_predicted["price"] = y_predicted

    df_predicted.iloc[0] = online_data[["index", "price"]].iloc[-1]
    df_predicted["type"] = "predicted"

    df_viz = pd.concat([online_data, df_predicted])
    fig = px.line(df_viz, x="index", y="price", color="type")

    st.write(fig)


'''
df = pd.read_csv("data/clean_historic_with_features/BTCUSDT-except2024-1d.csv")
df_2024 = pd.read_csv("data/clean_historic_with_features/BTCUSDT-2024-1d.csv")

df["type"] = "observed"
df_2024["type"] = "predicted"

df_streamlit = pd.concat([df, df_2024], ignore_index=True).reset_index()


for i in range(len(df_2024)):
    # st.line_chart(df_streamlit.iloc[i:], x="index", y="price", color="type")
    with plot_spot:
        update_chart(df_streamlit)

    df_streamlit.loc[(len(df) + i), "type"] = "observed"
'''
