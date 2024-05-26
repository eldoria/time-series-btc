import asyncio
import websockets
import csv
import json
import os


async def send_data(websocket, file='/data/clean_historic_with_features/BTCUSDT-1d.csv'):
    path_to_file = "/".join(os.path.abspath(__file__).split("/")[:-2]) + file
    with open(path_to_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            # Send each row from the CSV file to the client
            data = json.dumps({'index': index, 'data': ','.join(row)})
            await websocket.send(data)
        # Close the WebSocket connection after sending all data
        await websocket.close()

start_server = websockets.serve(send_data, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
