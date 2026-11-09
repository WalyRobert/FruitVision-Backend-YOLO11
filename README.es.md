# FruitVision — API experimental

Backend FastAPI para imágenes, vídeos cortos y frames por WebSocket. La versión incluye los módulos importados, validación de archivos, pruebas y un modo demo explícito sin inferencia ni detecciones inventadas.

Consulte [la documentación principal](README.md) para instalación, límites, autenticación, configuración de pesos YOLO locales y contratos. El modo YOLO requiere dependencias y pesos compatibles. La segmentación disponible es 2D; esta edición no implementa SAM 3D, reconstrucción 3D ni conteo de individuos únicos.

Las pruebas usan motores controlados y no certifican exactitud, FPS o GPU. Autor: Waly Robert.
