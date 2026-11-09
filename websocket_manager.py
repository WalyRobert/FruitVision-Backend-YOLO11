from fastapi import WebSocket


class ConnectionManager:
    def __init__(self, max_connections=4):
        self.active = set()
        self.max_connections = max_connections

    async def connect(self, websocket: WebSocket):
        if len(self.active) >= self.max_connections:
            await websocket.close(code=1013)
            return False
        await websocket.accept()
        self.active.add(websocket)
        return True

    def disconnect(self, websocket: WebSocket):
        self.active.discard(websocket)
