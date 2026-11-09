from detector import YOLODetector
from time import perf_counter


class YOLOSegmenter(YOLODetector):
    def segment(self, image):
        start = perf_counter()
        with self.lock:
            self._load()
            result = self.model.predict(source=image, verbose=False)[0]
            polygons = []
            if result.masks is not None and result.boxes is not None:
                for polygon, box in zip(result.masks.xy, result.boxes):
                    if (
                        not self.classes
                        or result.names[int(box.cls.item())] in self.classes
                    ):
                        polygons.append(polygon.tolist())
        return {
            "polygons": polygons,
            "num_objects": len(polygons),
            "processing_time": (perf_counter() - start) * 1000,
        }


class DemoSegmenter:
    model = None

    def segment(self, image):
        return {"polygons": [], "num_objects": 0, "processing_time": 0.0}

    def cleanup(self):
        pass
