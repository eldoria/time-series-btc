import pandas as pd
import os
import time

import streamlit as st
import plotly.express as px

path_to_real_time_folder = "/".join(os.path.abspath(__file__).split('/')[:-1])
folder = f"{path_to_real_time_folder}/data/real_time"
DATAS_FILE_PATH = f"{folder}/datas.csv"
PREDICTIONS_FILE_PATH = f"{folder}/predictions.csv"


speed_selector = st.select_slider("speed", options=["very slow", "slow", "medium", "fast", "very fast"])
if speed_selector == "very slow":
    sleep_time = 1
elif speed_selector == "slow":
    sleep_time = 0.5
if speed_selector == "medium":
    sleep_time = 0.3
if speed_selector == "fast":
    sleep_time = 0.1
if speed_selector == "very fast":
    sleep_time = 0.01


def get_data():
    counter = st.session_state["counter"]
    df1, df2 = data["price"].head(counter+1), predictions.iloc[counter].T.reset_index(drop=True).rename("price")
    df2.index = [index + 1 for index in df2.index]
    df2.iloc[0] = df1.iloc[-1]
    return df1, df2


def update_chart():
    df1, df2 = get_data()

    fig = px.line(df1, x=df1.index, y=df1)
    fig.add_scatter(x=[index + len(df1) - 2 for index in df2.index], y=df2, name="prediction")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    try:
        data = pd.read_csv(DATAS_FILE_PATH)
        predictions = pd.read_csv(PREDICTIONS_FILE_PATH)

        if "counter" not in st.session_state:
            st.session_state["counter"] = 0
        elif st.session_state["counter"] < (len(predictions)-1):
            st.session_state["counter"] += 1
            update_chart()
        else:
            pass
    except pd.errors.EmptyDataError:
        pass
    time.sleep(sleep_time)
    st.rerun()
