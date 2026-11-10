"""Diagnóstico local sem inferência, download de modelos ou leitura de credenciais."""
import importlib.util
import os
from pathlib import Path

root = Path(__file__).resolve().parent
mode = os.getenv("FRUITVISION_MODE", "demo")
print("FruitVision — diagnóstico local")
print("Modo:", mode, "(demo não faz inferência)" if mode == "demo" else "")
for name in ("fastapi", "uvicorn", "cv2", "numpy", "PIL"):
    print("OK:" if importlib.util.find_spec(name) else "PENDENTE:", name)
if mode == "yolo":
    for name in ("torch", "ultralytics"):
        print("OK:" if importlib.util.find_spec(name) else "PENDENTE:", name)
    for variable, default in (("YOLO_MODEL", "models/yolo11n.pt"), ("YOLO_SEGMENT_MODEL", "models/yolo11n-seg.pt")):
        file = Path(os.getenv(variable, default))
        if not file.is_absolute():
            file = root / file
        print("OK:" if file.is_file() else "PENDENTE:", variable, "(somente presença; carregamento não verificado)")
print("Autenticação configurada:", bool(os.getenv("FRUITVISION_API_KEY", "").strip()))
print("Nenhum modelo foi baixado ou executado. Consulte /health após iniciar app.py.")
