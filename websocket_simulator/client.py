import asyncio
import websockets
import json
import subprocess
import time

from models.xgb_regressor import *

xgb_regressor = XgbRegressor()
n_steps_prediction = 30

path_to_real_time_folder = "/".join(os.path.abspath(__file__).split('/')[:-2])
folder = f"{path_to_real_time_folder}/data/real_time"
DATAS_FILE_PATH = f"{folder}/datas.csv"
PREDICTIONS_FILE_PATH = f"{folder}/predictions.csv"

with open(DATAS_FILE_PATH, 'w') as file1:
    pass

with open(PREDICTIONS_FILE_PATH, 'w') as file2:
    pass


async def append_dataframe_to_csv(dataframe, csv_file_path):
    file_size = os.path.getsize(csv_file_path)
    if file_size == 0:
        dataframe.to_csv(csv_file_path, mode='a', header=True, index=False)
    else:
        dataframe.to_csv(csv_file_path, mode='a', header=False, index=False)


async def receive_data():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri, ping_timeout=60, ping_interval=10) as websocket:
        path_to_script = "/".join(os.path.abspath(__file__).split('/')[:-2]) + "/streamlit_realtime_prediction.py"
        subprocess.Popen(
            ["streamlit", "run", path_to_script],
            stdout=open('/dev/null', 'w'),
            stderr=open('logfile.log', 'a'),
            preexec_fn=os.setpgrp
        )
        while True:
            # Receive data from the server
            data = await asyncio.wait_for(websocket.recv(), timeout=300)
            data = json.loads(data)
            df = xgb_regressor.json_to_df(data)

            if df is None:
                continue

            await append_dataframe_to_csv(df, DATAS_FILE_PATH)

            xgb_regressor.online_learning(df)
            df = pd.read_csv(DATAS_FILE_PATH, dtype=xgb_regressor.dtype)

            predictions = xgb_regressor.predict(df, n_steps_prediction)
            predictions_reformatted = xgb_regressor.prediction_rows_to_cols(predictions)

            await append_dataframe_to_csv(predictions_reformatted, PREDICTIONS_FILE_PATH)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(receive_data())
