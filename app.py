"""FruitVision: API local de detecção e segmentação 2D, com demonstração explícita."""

import asyncio
import base64
import binascii
import hmac
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from detector import YOLODetector, DemoDetector, ModelUnavailable
from segmenter import YOLOSegmenter, DemoSegmenter
from websocket_manager import ConnectionManager
from media import decode_image, bounded_read
from video import process_video_bytes
from models import DetectionResponse, SegmentationResponse, VideoProcessingResponse
from upload_limit import UploadLimitMiddleware
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


def create_app(detector=None, segmenter=None, output_dir=None):
    mode = os.getenv("FRUITVISION_MODE", "demo")
    if mode not in ("demo", "yolo"):
        raise ValueError("FRUITVISION_MODE deve ser demo ou yolo.")
    classes = [
        x.strip()
        for x in os.getenv("FRUIT_CLASSES", "apple,banana,orange").split(",")
        if x.strip()
    ]
    detect_engine = (
        detector
        if detector is not None
        else (
            DemoDetector()
            if mode == "demo"
            else YOLODetector(os.getenv("YOLO_MODEL", "models/yolo11n.pt"), classes)
        )
    )
    segment_engine = (
        segmenter
        if segmenter is not None
        else (
            DemoSegmenter()
            if mode == "demo"
            else YOLOSegmenter(
                os.getenv("YOLO_SEGMENT_MODEL", "models/yolo11n-seg.pt"), classes
            )
        )
    )
    output = Path(output_dir or os.getenv("OUTPUT_DIR", "data/outputs")).resolve()
    upload_limit = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
    max_pixels = int(os.getenv("MAX_IMAGE_PIXELS", "12000000"))
    max_outputs = int(os.getenv("MAX_OUTPUT_FILES", "20"))
    key = os.getenv("FRUITVISION_API_KEY", "")
    origins = [
        s.strip()
        for s in os.getenv("ALLOWED_ORIGINS", os.getenv("FRUITVISION_CORS_ORIGINS", "http://127.0.0.1:5173")).split(",")
        if s.strip()
    ]
    gate = asyncio.Lock()
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app):
        output.mkdir(parents=True, exist_ok=True)
        yield
        detect_engine.cleanup()
        segment_engine.cleanup()

    api = FastAPI(title="FruitVision", version="0.2.0", lifespan=lifespan)
    api.add_middleware(UploadLimitMiddleware, max_bytes=upload_limit + 65536)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    async def authorize(x_api_key: str | None = Header(default=None)):
        if key and not hmac.compare_digest(x_api_key or "", key):
            raise HTTPException(401, "Chave de acesso inválida.")

    async def run_job(function, *args):
        if gate.locked():
            raise HTTPException(
                429,
                "Processamento ocupado. Tente novamente após concluir a tarefa atual.",
            )
        async with gate:
            try:
                return await run_in_threadpool(function, *args)
            except ModelUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Falha de processamento")
                raise HTTPException(
                    500, "Falha de processamento. Consulte os logs do servidor."
                ) from exc

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "mode": mode,
            "inference_enabled": mode == "yolo",
            "detector_loaded": getattr(detect_engine, "model", None) is not None,
            "segmenter_loaded": getattr(segment_engine, "model", None) is not None,
            "segmentation": "demo" if mode == "demo" else "yolo-2d",
            "authentication_required": bool(key),
        }

    def image_job(data, segment=False):
        image = decode_image(data, max_pixels)
        if segment:
            result = segment_engine.segment(image)
            return SegmentationResponse(
                mode=mode,
                engine="demo" if mode == "demo" else "yolo-2d",
                polygons=result["polygons"],
                num_objects=result["num_objects"],
                processing_time_ms=result["processing_time"],
            )
        result = detect_engine.detect(image)
        return DetectionResponse(
            mode=mode,
            detections=result["detections"],
            image_shape=(image.shape[1], image.shape[0]),
            processing_time_ms=result["processing_time"],
        )

    @api.post(
        "/detect", response_model=DetectionResponse, dependencies=[Depends(authorize)]
    )
    async def detect(file: UploadFile = File(...)):
        return await run_job(image_job, await bounded_read(file, upload_limit))

    @api.post(
        "/segment",
        response_model=SegmentationResponse,
        dependencies=[Depends(authorize)],
    )
    async def segment(file: UploadFile = File(...)):
        return await run_job(image_job, await bounded_read(file, upload_limit), True)

    def video_job(data, enabled):
        output.mkdir(parents=True, exist_ok=True)
        for old in output.glob("*.mp4"):
            if time() - old.stat().st_mtime > 3600:
                old.unlink(missing_ok=True)
        if len(list(output.glob("*.mp4"))) >= max_outputs:
            raise HTTPException(
                507, "Armazenamento temporário cheio. Aguarde a expiração dos arquivos."
            )
        result = process_video_bytes(
            data,
            detect_engine,
            segment_engine,
            output,
            enabled,
            int(os.getenv("MAX_VIDEO_FRAMES", "300")),
            float(os.getenv("MAX_VIDEO_SECONDS", "30")),
            max_pixels,
        )
        return VideoProcessingResponse(mode=mode, **result)

    @api.post(
        "/process-video",
        response_model=VideoProcessingResponse,
        dependencies=[Depends(authorize)],
    )
    async def process_video(
        file: UploadFile = File(...), enable_segmentation: bool = Form(False)
    ):
        return await run_job(
            video_job, await bounded_read(file, upload_limit), enable_segmentation
        )

    @api.get("/videos/{video_id}", dependencies=[Depends(authorize)])
    async def download(video_id: str):
        if not re.fullmatch("[a-f0-9]{32}", video_id):
            raise HTTPException(404, "Vídeo não encontrado.")
        path = output / (video_id + ".mp4")
        if not path.is_file() or time() - path.stat().st_mtime > 3600:
            raise HTTPException(404, "Vídeo ausente ou expirado.")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @api.websocket("/ws")
    async def websocket(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin and origin not in origins:
            await ws.close(code=1008)
            return
        if key and not hmac.compare_digest(ws.headers.get("x-api-key", ""), key):
            # Clientes de navegador podem usar a primeira mensagem para autenticar.
            await ws.accept()
            try:
                auth = await asyncio.wait_for(ws.receive_text(), timeout=5)
                credentials = json.loads(auth) if len(auth) <= 4096 else None
                if (
                    not isinstance(credentials, dict)
                    or not isinstance(credentials.get("api_key"), str)
                    or not hmac.compare_digest(credentials["api_key"], key)
                ):
                    await ws.close(code=1008)
                    return
            except (ValueError, asyncio.TimeoutError, WebSocketDisconnect):
                await ws.close(code=1008)
                return
            if len(manager.active) >= manager.max_connections:
                await ws.close(code=1013)
                return
            manager.active.add(ws)
        elif not await manager.connect(ws):
            return
        try:
            await ws.send_json({"type": "ready", "mode": mode})
            while True:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
                if len(raw) > 2_000_000:
                    await ws.close(code=1009)
                    return
                try:
                    packet = json.loads(raw)
                    if (
                        not isinstance(packet, dict)
                        or packet.get("type") != "frame"
                        or not isinstance(packet.get("frame"), str)
                    ):
                        raise ValueError(
                            "Use type=frame e frame com bytes codificados."
                        )
                    data = (
                        base64.b64decode(packet["frame"], validate=True)
                        if packet.get("encoding") == "base64"
                        else bytes.fromhex(packet["frame"])
                    )
                    result = await run_job(image_job, data)
                    await ws.send_json(
                        {"type": "detection_results", **result.model_dump()}
                    )
                except (ValueError, binascii.Error, HTTPException) as exc:
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": exc.detail
                            if isinstance(exc, HTTPException)
                            else "Mensagem ou frame inválido.",
                        }
                    )
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            manager.disconnect(ws)
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()

    return api


app = create_app()
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, ws_max_size=2_000_000)
