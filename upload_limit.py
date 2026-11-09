"""Bound multipart traffic before Starlette spools uploaded files to disk."""

from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException


class PayloadTooLarge(HTTPException):
    def __init__(self):
        super().__init__(413, "Corpo da solicitação muito grande.")


class UploadLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Content-Length inválido."}, 400)(
                scope, receive, send
            )
        if declared > self.max_bytes:
            return await JSONResponse(
                {"detail": "Corpo da solicitação muito grande."}, 413
            )(scope, receive, send)
        size = 0
        started = False

        async def bounded_receive():
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    raise PayloadTooLarge()
            return message

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, tracked_send)
        except PayloadTooLarge:
            if not started:
                await JSONResponse(
                    {"detail": "Corpo da solicitação muito grande."}, 413
                )(scope, receive, send)
