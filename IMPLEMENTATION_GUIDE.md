# FruitVision Backend - Guia de Implementação Completo

## 📋 Visão Geral

Este guia descreve como usar o backend FruitVision com YOLO11, ObjectCounter, Meta SAM 3D, API REST, e WebSocket para detecção e segmentação de frutas em tempo real.

## 🚀 Quick Start

### 1. Instalação de Dependências

```bash
# Clone o repositório
git clone https://github.com/WalyRobert/FruitVision-Backend-YOLO11.git
cd FruitVision-Backend-YOLO11

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 2. Iniciar o Server

```bash
python app.py
# Server estará em: http://localhost:8000
```

## 📡 API REST Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Detecção de Frutas (Imagem)
```bash
POST http://localhost:8000/detect
Body: multipart/form-data
  - file: <image.jpg>

Response:
{
  "success": true,
  "detections": [
    {
      "id": 0,
      "class": "apple",
      "confidence": 0.95,
      "bbox": [100, 150, 250, 350]
    }
  ],
  "image_shape": [640, 480],
  "processing_time_ms": 45.2
}
```

### Segmentação com SAM 3D
```bash
POST http://localhost:8000/segment
Body: multipart/form-data
  - file: <image.jpg>

Response:
{
  "success": true,
  "masks": [...],
  "num_objects": 5,
  "processing_time_ms": 120.5
}
```

### Processamento de Vídeo
```bash
POST http://localhost:8000/process-video
Body: multipart/form-data
  - file: <video.mp4>
  - enable_segmentation: true/false

Response:
{
  "success": true,
  "output_file": "output_video.mp4",
  "total_frames": 300,
  "total_detections": 1245,
  "fps": 30,
  "resolution": [1920, 1080]
}
```

## 🔗 WebSocket em Tempo Real

### Conectar e Enviar Frames

```javascript
// JavaScript/React
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected');
  // Enviar frame de vídeo
  const frameData = canvasToHex(videoFrame);
  ws.send(JSON.stringify({
    type: 'frame',
    frame: frameData
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Detections:', result.detections);
  console.log('Processing time:', result.processing_time_ms, 'ms');
};
```

```python
# Python
import asyncio
import websockets
import cv2
import base64

async def stream_video():
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        cap = cv2.VideoCapture(0)  # Webcam
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_hex = buffer.tobytes().hex()
            
            # Send to server
            await ws.send(json.dumps({
                'type': 'frame',
                'frame': frame_hex
            }))
            
            # Receive results
            response = await ws.recv()
            results = json.loads(response)
            print(f"Detections: {results['detections']}")
            print(f"FPS: {1000/results['processing_time_ms']:.1f}")

asyncio.run(stream_video())
```

## 📊 Estrutura do Projeto

```
FruitVision-Backend-YOLO11/
├── app.py                    # FastAPI principal
├── models.py                 # Modelos Pydantic
├── detector.py              # Classe YOLODetector com ObjectCounter
├── segmenter.py             # Classe SAM3DSegmenter
├── websocket_manager.py      # Gerenciador de conexões WebSocket
├── video_processor.py       # Processador de vídeos
├── requirements.txt         # Dependências Python
├── .env.example             # Variáveis de ambiente
└── README.md                # Documentação
```

## 🔧 Classe YOLODetector (detector.py)

```python
from detector import YOLODetector

# Inicializar detector
detector = YOLODetector(model_name="yolo11n")  # Opções: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

# Detectar frutas em imagem
import cv2
image = cv2.imread('fruit.jpg')
results = detector.detect(image)

print(results)
# {
#   'detections': [
#     {'class': 'apple', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]},
#     {'class': 'orange', 'confidence': 0.92, 'bbox': [x1, y1, x2, y2]}
#   ],
#   'processing_time': 45.2  # ms
# }
```

## 🔍 Classe SAM3DSegmenter (segmenter.py)

```python
from segmenter import SAM3DSegmenter

# Inicializar segmentador
segmenter = SAM3DSegmenter()

# Segmentar frutas
import cv2
image = cv2.imread('fruit.jpg')
results = segmenter.segment(image)

print(results)
# {
#   'masks': [binary_mask_1, binary_mask_2, ...],
#   'num_objects': 5,
#   'processing_time': 120.5  # ms
# }
```

## 📱 Conectar com React Frontend

### Exemplo React Hook

```javascript
import { useEffect, useRef, useState } from 'react';

function FruitDetection() {
  const wsRef = useRef(null);
  const [detections, setDetections] = useState([]);
  const videoRef = useRef(null);

  useEffect(() => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    
    wsRef.current.onmessage = (event) => {
      const { detections, processing_time_ms } = JSON.parse(event.data);
      setDetections(detections);
      console.log(`FPS: ${(1000 / processing_time_ms).toFixed(1)}`);
    };
    
    return () => wsRef.current?.close();
  }, []);

  const sendFrame = () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    
    const frameData = canvas.toDataURL('image/jpeg').split(',')[1];
    wsRef.current.send(JSON.stringify({
      type: 'frame',
      frame: Buffer.from(frameData, 'base64').toString('hex')
    }));
  };

  return (
    <div>
      <video ref={videoRef} autoPlay></video>
      <button onClick={sendFrame}>Detect Fruits</button>
      <div>
        {detections.map((det, i) => (
          <p key={i}>{det.class}: {(det.confidence * 100).toFixed(1)}%</p>
        ))}
      </div>
    </div>
  );
}
```

## 🎯 Recursos Implementados

✅ **YOLO11 Detection**
- Detecção de frutas em tempo real
- 5 tamanhos de modelo (nano até xlarge)
- Suporte para GPU/CPU

✅ **ObjectCounter (Ultralytics)**
- Contagem de objetos
- Rastreamento com ID único
- Histórico de detecções

✅ **Meta SAM 3D**
- Segmentação precisa de frutas
- Máscaras de qualidade alta
- Processamento em tempo real

✅ **API REST (FastAPI)**
- Upload de imagens e vídeos
- Processamento assíncrono
- Respostas em JSON

✅ **WebSocket**
- Streaming de vídeo em tempo real
- Resposta de detecção baixa latência
- Múltiplas conexões simultâneas

✅ **Processamento de Vídeo**
- Upload de vídeos MP4
- Processamento frame-by-frame
- Output vídeo com anotações

## 🔐 Configuração de Variáveis de Ambiente

Crie um arquivo `.env`:

```env
API_HOST=0.0.0.0
API_PORT=8000
YOLO_MODEL=yolo11m
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
GPU_DEVICE=0
MAX_WORKERS=4
LOG_LEVEL=INFO
```

## 📈 Performance

| Modelo | Detecções/s | Latência | GPU Memory |
|--------|------------|----------|------------|
| YOLOv11n | 120+ | ~8ms | 2GB |
| YOLOv11s | 80+ | ~12ms | 3GB |
| YOLOv11m | 50+ | ~20ms | 5GB |
| YOLOv11l | 30+ | ~33ms | 8GB |

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"
```bash
# Use modelo menor
export YOLO_MODEL=yolo11n
# Ou use CPU
export DEVICE=cpu
```

### WebSocket timeout
```python
# Aumente timeout no cliente
ws = websockets.connect('ws://...', ping_interval=20, ping_timeout=10)
```

## 📞 Suporte

Para dúvidas ou bugs, abra uma issue no GitHub.

---

**Desenvolvido com ❤️ para detecção de frutas em tempo real**
