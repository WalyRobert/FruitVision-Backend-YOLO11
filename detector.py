from pathlib import Path
from threading import Lock
from time import perf_counter


class ModelUnavailable(RuntimeError):
    pass


class YOLODetector:
    def __init__(self, model_name, classes=None):
        self.model_path = Path(model_name)
        self.model = None
        self.lock = Lock()
        self.classes = set(classes or ["apple", "banana", "orange"])

    def _load(self):
        if self.model is None:
            if not self.model_path.is_file():
                raise ModelUnavailable(
                    "Pesos YOLO locais ausentes. Configure YOLO_MODEL e instale requirements-ml.txt."
                )
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.model_path))
            except Exception as exc:
                raise ModelUnavailable(
                    "Não foi possível carregar YOLO. Verifique dependências e pesos locais."
                ) from exc

    def detect(self, image):
        start = perf_counter()
        with self.lock:
            self._load()
            result = self.model.predict(source=image, verbose=False)[0]
            items = []
            if result.boxes is not None:
                for box in result.boxes:
                    cid = int(box.cls.item())
                    label = result.names[cid]
                    if self.classes and label not in self.classes:
                        continue
                    items.append(
                        {
                            "class_id": cid,
                            "label": label,
                            "confidence": float(box.conf.item()),
                            "bbox": box.xyxy[0].tolist(),
                        }
                    )
        return {"detections": items, "processing_time": (perf_counter() - start) * 1000}

    def cleanup(self):
        self.model = None


class DemoDetector:
    model = None

    def detect(self, image):
        # Deliberadamente vazio: modo de demonstração não inventa inferências.
        return {"detections": [], "processing_time": 0.0}

    def cleanup(self):
        pass
