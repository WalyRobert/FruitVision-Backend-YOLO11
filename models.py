from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


class DetectionResponse(BaseModel):
    success: bool = True
    mode: str
    detections: list[Detection]
    image_shape: tuple[int, int]
    processing_time_ms: float


class SegmentationResponse(BaseModel):
    success: bool = True
    mode: str
    engine: str
    polygons: list[list[list[float]]]
    num_objects: int
    processing_time_ms: float


class VideoProcessingResponse(BaseModel):
    success: bool = True
    mode: str
    output_file: str
    download_url: str
    total_frames: int
    total_detections: int
    fps: float
    resolution: tuple[int, int]
    segmentation_enabled: bool
