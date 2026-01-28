import asyncio
import websockets
import json

async def handler(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            print(f"Received: {message}")
            await websocket.send(json.dumps({"status": "ack"}))
    except Exception as e:
        print(f"Error: {e}")

async def main(port):
    print(f"Server starting on port {port}")
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"Listening on port {port}")
        await asyncio.Future()  # run forever

def run(port):
    asyncio.run(main(port))
