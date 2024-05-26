import pandas as pd
import streamlit as st
import time
import math

from main import y_train, y_test, y_predicted


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

df = pd.DataFrame()
df["predicted"] = y_train
df["observed"] = y_train

chart = st.line_chart(df, y=["predicted", "observed"])


for i in range(len(y_test)):
    percentage_completion = i / len(y_test)

    observed = y_test.iloc[i]
    predicted = int(y_predicted.iloc[i])

    status_text.text("%i%% Complete" % math.ceil(percentage_completion*100))

    new_row = pd.DataFrame()
    new_row["predicted"] = pd.Series(predicted)
    new_row["observed"] = pd.Series(observed)

    chart.add_rows(new_row)
    progress_bar.progress(percentage_completion)

    time.sleep(0.005)


progress_bar.empty()
st.button("Re-run")
