import io
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import create_app
from detector import ModelUnavailable, YOLODetector


class FakeDetector:
    model = None

    def detect(self, image):
        return {
            "detections": [
                {
                    "class_id": 0,
                    "label": "apple",
                    "confidence": 0.9,
                    "bbox": [2, 2, 12, 12],
                }
            ],
            "processing_time": 1,
        }

    def cleanup(self):
        pass


class FakeSegmenter:
    model = None

    def segment(self, image):
        return {
            "polygons": [[[1, 1], [10, 1], [10, 10]]],
            "num_objects": 1,
            "processing_time": 1,
        }

    def cleanup(self):
        pass


def png():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 30, 30)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUITVISION_MODE", "demo")
    monkeypatch.delenv("FRUITVISION_API_KEY", raising=False)
    with TestClient(
        create_app(FakeDetector(), FakeSegmenter(), tmp_path / "outputs")
    ) as c:
        yield c


def test_detect_validates_and_returns_schema(client):
    assert client.get("/health").json()["mode"] == "demo"
    response = client.post("/detect", files={"file": ("x.png", png(), "image/png")})
    assert response.status_code == 200
    assert response.json()["detections"][0]["label"] == "apple"
    assert response.json()["image_shape"] == [32, 32]
    assert (
        client.post("/detect", files={"file": ("bad.png", b"garbage")}).status_code
        == 400
    )
    assert client.post("/detect", files={"file": ("empty.png", b"")}).status_code == 400


def test_segment_and_websocket(client):
    assert (
        client.post("/segment", files={"file": ("x.png", png())}).json()["num_objects"]
        == 1
    )
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "frame", "frame": png().hex()})
        assert ws.receive_json()["detections"][0]["label"] == "apple"
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "frame", "frame": "xyz"})
        assert ws.receive_json()["type"] == "error"


def test_video_is_written_and_downloadable(client, tmp_path):
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((32, 32, 3), 150, dtype=np.uint8))
    writer.release()
    response = client.post(
        "/process-video",
        files={"file": ("../../escape.avi", source.read_bytes(), "video/avi")},
        data={"enable_segmentation": "true"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_frames"] == 3 and data["total_detections"] == 3
    assert data["segmentation_enabled"] is True
    output = client.get(data["download_url"])
    assert output.status_code == 200 and len(output.content) > 0
    saved = tmp_path / "verified.mp4"
    saved.write_bytes(output.content)
    cap = cv2.VideoCapture(str(saved))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (32, 32)
    assert not (tmp_path / "escape.avi").exists()
    assert (
        client.post(
            "/process-video", files={"file": ("bad.mp4", b"invalid")}
        ).status_code
        == 400
    )


def test_auth_limits_and_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUITVISION_API_KEY", "test-key")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "100")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    with TestClient(create_app(output_dir=tmp_path)) as c:
        assert c.post("/detect", files={"file": ("x.png", png())}).status_code == 401
        headers = {"X-API-Key": "test-key"}
        assert (
            c.post(
                "/detect", headers=headers, files={"file": ("x.png", png())}
            ).status_code
            == 413
        )
        assert (
            c.post(
                "/detect", headers=headers, files={"file": ("x", b"x" * 1_048_577)}
            ).status_code
            == 413
        )
    detector = YOLODetector(tmp_path / "missing.pt")
    with pytest.raises(ModelUnavailable):
        detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))


def test_demo_does_not_fabricate_detections(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUITVISION_MODE", "demo")
    monkeypatch.delenv("FRUITVISION_API_KEY", raising=False)
    with TestClient(create_app(output_dir=tmp_path)) as c:
        result = c.post("/detect", files={"file": ("x.png", png())}).json()
        assert result["mode"] == "demo" and result["detections"] == []


def test_video_limits_remove_partial_output(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_VIDEO_FRAMES", "1")
    source = tmp_path / "long.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert (
        client.post(
            "/process-video", files={"file": ("long.avi", source.read_bytes())}
        ).status_code
        == 413
    )
    assert list((tmp_path / "outputs").glob("*.mp4")) == []


def test_health_remains_responsive_and_parallel_jobs_are_rejected(
    tmp_path, monkeypatch
):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    started = threading.Event()

    class SlowDetector(FakeDetector):
        def detect(self, image):
            started.set()
            time.sleep(0.5)
            return super().detect(image)

    monkeypatch.delenv("FRUITVISION_API_KEY", raising=False)
    with TestClient(create_app(SlowDetector(), FakeSegmenter(), tmp_path)) as c:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                lambda: c.post("/detect", files={"file": ("x.png", png())})
            )
            assert started.wait(5)
            assert c.get("/health").status_code == 200
            assert (
                c.post("/detect", files={"file": ("x.png", png())}).status_code == 429
            )
            assert future.result().status_code == 200


def test_websocket_authentication_and_invalid_packet_shapes(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUITVISION_API_KEY", "test-key")
    with TestClient(create_app(output_dir=tmp_path)) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json({"api_key": "test-key"})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json([])
            assert ws.receive_json()["type"] == "error"
        with pytest.raises(Exception) as result:
            with c.websocket_connect("/ws") as ws:
                ws.send_json([])
                ws.receive_json()
        assert getattr(result.value, "code", None) == 1008


def test_http_rejects_missing_weights_and_oversized_stream(tmp_path, monkeypatch):
    monkeypatch.delenv("FRUITVISION_API_KEY", raising=False)
    monkeypatch.setenv("FRUITVISION_MODE", "yolo")
    monkeypatch.setenv("YOLO_MODEL", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    with TestClient(create_app(output_dir=tmp_path)) as c:
        assert c.post("/detect", files={"file": ("x.png", png())}).status_code == 503
        boundary = "test-boundary"
        start = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
        body = iter([start, b"x" * 1_200_000, f"\r\n--{boundary}--\r\n".encode()])
        assert (
            c.post(
                "/detect",
                content=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            ).status_code
            == 413
        )
