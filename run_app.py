import pandas as pd
import streamlit as st
import numpy as np
import time

from main import y_train, y_test, y_predicted


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

df = pd.DataFrame()
df["predicted"] = y_train
df["observed"] = y_train


chart = st.line_chart(df, y=["predicted", "observed"])
# chart = st.line_chart([[val, val] for val in y_train.values])#, y=["observed", "predicted"])


for i in range(len(y_test)):
    percentage_completion = i / len(y_test)

    observed = y_test.iloc[i]
    predicted = int(y_predicted.iloc[i])

    status_text.text("%i%% Complete" % (percentage_completion*100))
    # chart.add_rows(np.column_stack((predicted, observed)))

    new_row = pd.DataFrame()
    new_row["predicted"] = pd.Series(predicted)
    new_row["observed"] = pd.Series(observed)

    chart.add_rows(new_row)
    progress_bar.progress(percentage_completion)

    time.sleep(0.01)

progress_bar.empty()
st.button("Re-run")

'''
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows1 = np.random.randn(1, 1)
last_rows2 = np.random.randn(1, 1)
chart = st.line_chart(np.column_stack((last_rows1, last_rows2)), columns=["a", "b"])


for i in range(1, 101):
    new_rows1 = last_rows1[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    new_rows2 = last_rows2[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(np.column_stack((new_rows1, new_rows2)))
    progress_bar.progress(i)
    last_rows1 = new_rows1
    last_rows2 = new_rows2
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
'''