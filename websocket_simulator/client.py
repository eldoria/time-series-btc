import asyncio
import websockets
import json
import subprocess
import os

from streamlit_realtime_prediction import update_chart


async def receive_data():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        '''
        path_to_streamlit_file = "/".join(os.path.abspath(__file__).split("/")[:-2]) + "/streamlit_realtime_prediction.py"
        print(os.path.abspath(__file__).split("/")[:-1])
        print(path_to_streamlit_file)
        subprocess.Popen(["streamlit", "run", path_to_streamlit_file])
        '''
        while True:
            # Receive data from the server
            print("loop")
            data = await websocket.recv()
            data = json.loads(data)
            update_chart(data)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(receive_data())
