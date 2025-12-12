#!/usr/bin/env python3
"""
FruitVision Backend API
FastAPI application with YOLO11 object detection,
ObjectCounter tracking, SAM 3D segmentation,
and WebSocket streaming support.
"""

import os
import io
from pathlib import Path
from typing import Optional
import asyncio
import json

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter

from models import DetectionResponse, VideoProcessingResponse, SegmentationResponse
from detector import YOLODetector
from segmenter import SAM3DSegmenter
from websocket_manager import ConnectionManager

# Initialize FastAPI app
app = FastAPI(
    title="FruitVision Backend API",
    description="Real-time fruit detection, tracking, and segmentation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global managers and detectors
manager = ConnectionManager()
detector: Optional[YOLODetector] = None
segmenter: Optional[SAM3DSegmenter] = None

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global detector, segmenter
    print("[STARTUP] Loading YOLO11 detector...")
    detector = YOLODetector(model_name="yolo11n")
    
    print("[STARTUP] Loading SAM 3D segmenter...")
    segmenter = SAM3DSegmenter()
    
    print("[STARTUP] Models loaded successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("[SHUTDOWN] Cleaning up resources...")
    if detector:
        detector.cleanup()
    if segmenter:
        segmenter.cleanup()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_loaded": detector is not None,
        "segmenter_loaded": segmenter is not None
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_fruits(file: UploadFile = File(...)):
    """Detect fruits in a single image"""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = detector.detect(image)
        
        return DetectionResponse(
            success=True,
            detections=results["detections"],
            image_shape=(image.shape[1], image.shape[0]),
            processing_time_ms=results["processing_time"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/segment", response_model=SegmentationResponse)
async def segment_fruits(file: UploadFile = File(...)):
    """Segment fruits in an image using SAM 3D"""
    if not segmenter:
        raise HTTPException(status_code=503, detail="Segmenter not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run segmentation
        results = segmenter.segment(image)
        
        return SegmentationResponse(
            success=True,
            masks=results["masks"],
            num_objects=results["num_objects"],
            processing_time_ms=results["processing_time"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process-video", response_model=VideoProcessingResponse)
async def process_video(file: UploadFile = File(...), enable_segmentation: bool = False):
    """Process a video file with YOLO detection and optional SAM 3D segmentation"""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not loaded")
    
    try:
        # Save uploaded file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process video
        output_path = f"output_{file.filename}"
        
        cap = cv2.VideoCapture(temp_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        counter = ObjectCounter()
        frame_count = 0
        detections_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection
            results = detector.detect(frame)
            detections_list.extend(results["detections"])
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        os.remove(temp_path)
        
        return VideoProcessingResponse(
            success=True,
            output_file=output_path,
            total_frames=frame_count,
            total_detections=len(detections_list),
            fps=fps,
            resolution=(width, height)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "frame":
                # Decode frame
                frame_data = data["frame"]
                nparr = np.frombuffer(bytes.fromhex(frame_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Run detection
                results = detector.detect(frame)
                
                # Send back results
                response = {
                    "type": "detection_results",
                    "detections": results["detections"],
                    "processing_time_ms": results["processing_time"]
                }
                
                await websocket.send_json(response)
    
    except Exception as e:
        await manager.disconnect(websocket)
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
