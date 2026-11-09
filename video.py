from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from uuid import uuid4
import cv2
import numpy as np
from fastapi import HTTPException


def process_video_bytes(
    data,
    detector,
    segmenter,
    output_dir,
    enable_segmentation,
    max_frames,
    max_seconds,
    max_pixels,
):
    output = Path(output_dir) / (uuid4().hex + ".mp4")
    cap = None
    writer = None
    success = False
    temp = TemporaryDirectory(prefix="fruitvision-")
    try:
        source = Path(temp.name) / "input.bin"
        source.write_bytes(data)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise HTTPException(400, "Vídeo inválido ou codec não suportado.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or not 0 < fps <= 240 or width <= 0 or height <= 0:
            raise HTTPException(400, "Metadados de vídeo inválidos.")
        if width * height > max_pixels:
            raise HTTPException(413, "Vídeo excede o limite de pixels por frame.")
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise HTTPException(503, "Codec de gravação indisponível.")
        frames = 0
        total = 0
        start = monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (
                frames >= max_frames
                or frames / fps >= max_seconds
                or monotonic() - start > max_seconds
            ):
                raise HTTPException(
                    413,
                    "Vídeo excede o limite de frames, duração ou processamento. Envie um trecho menor.",
                )
            detections = detector.detect(frame)["detections"]
            total += len(detections)
            if enable_segmentation:
                for polygon in segmenter.segment(frame)["polygons"]:
                    cv2.polylines(
                        frame,
                        [np.asarray(polygon, dtype=np.int32)],
                        True,
                        (0, 200, 255),
                        2,
                    )
            for item in detections:
                x1, y1, x2, y2 = map(int, item["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.putText(
                    frame,
                    item["label"],
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 80),
                    1,
                )
            writer.write(frame)
            frames += 1
        if not frames:
            raise HTTPException(400, "Vídeo sem frames decodificáveis.")
        writer.release()
        writer = None
        cap.release()
        cap = None
        if not output.is_file() or output.stat().st_size == 0:
            raise HTTPException(500, "Falha ao gravar vídeo.")
        success = True
        return {
            "output_file": output.name,
            "download_url": "/videos/" + output.stem,
            "total_frames": frames,
            "total_detections": total,
            "fps": fps,
            "resolution": (width, height),
            "segmentation_enabled": enable_segmentation,
        }

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not success:
            output.unlink(missing_ok=True)
        temp.cleanup()
