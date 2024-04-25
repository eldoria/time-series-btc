import asyncio
import websockets
import csv


async def send_data(websocket, path):
    with open('../data/clean_historic/BTCUSDT-1d-2024.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Send each row from the CSV file to the client
            await websocket.send(','.join(row))
            await asyncio.sleep(1)
        # Close the WebSocket connection after sending all data
        await websocket.close()

start_server = websockets.serve(send_data, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
