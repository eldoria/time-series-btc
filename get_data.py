import websocket
import json


def on_message(ws, message):
    # ignore canddlestick data that is not closed
    if not json.loads(message)['k']['x']:
        return
    print(message)
    with open("data/real_time/data.txt", "a") as output_file:
        output_file.write(message + "\n")

def on_error(ws, error): print(error)
def on_close(wd): print("### closed ###")


def stream_kline(symbol, interval):
    socket = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    ws = websocket.WebSocketApp(socket,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()


stream_kline("btcusdt", "1m")
