[Português](README.md) · [Español](README.es.md)

# FruitVision — Backend de visión artificial

**Idioma:** Español  
**Estado:** Prototipo experimental. La API está iniciada, pero la estructura actual necesita módulos y validaciones antes de considerarse ejecutable.

FruitVision es un proyecto experimental de backend para analizar imágenes y vídeos mediante visión artificial. La propuesta explora la detección de objetos, el conteo/seguimiento y la segmentación, con una API que pueda conectarse a una interfaz web.

## Objetivos

- Recibir imágenes para detección.
- Recibir imágenes para segmentación.
- Procesar vídeos.
- Proporcionar resultados mediante HTTP y WebSocket.
- Evolucionar hacia la integración con una interfaz React.

## Tecnologías identificadas

- Python
- FastAPI y Uvicorn
- OpenCV y NumPy
- Ultralytics / YOLO
- WebSocket
- PyTorch y bibliotecas de visión artificial incluidas en `requirements.txt`

Las dependencias enumeradas representan la configuración prevista y todavía deben validarse en un entorno limpio y compatible.

## Endpoints presentes en `app.py`

| Método | Ruta | Finalidad descrita |
|---|---|---|
| GET | `/health` | Comprobar el estado del servicio |
| POST | `/detect` | Recibir una imagen para detección |
| POST | `/segment` | Recibir una imagen para segmentación |
| POST | `/process-video` | Iniciar el procesamiento de vídeo |
| WebSocket | `/ws` | Recibir fotogramas y devolver resultados de detección |

## Módulos referenciados

El archivo principal importa:

- `models`: modelos de respuesta de la API.
- `detector`: encapsulación de la detección YOLO.
- `segmenter`: encapsulación de la segmentación.
- `websocket_manager`: gestión de conexiones WebSocket.

**Pendiente de reproducibilidad:** estos módulos importados no se localizaron en la revisión del contenido actual del repositorio. La aplicación puede fallar al iniciarse hasta que se añadan los archivos o se corrijan las importaciones.

## Instalación prevista

```bash
git clone https://github.com/WalyRobert/FruitVision-Backend-YOLO11.git
cd FruitVision-Backend-YOLO11
python -m venv .venv
```

Activa el entorno virtual y revisa la compatibilidad de las dependencias antes de instalarlas:

```bash
pip install -r requirements.txt
```

Comando de inicio previsto:

```bash
uvicorn app:app --reload
```

**Atención:** la ejecución todavía no ha sido validada y puede fallar debido a módulos ausentes o incompatibilidades entre dependencias.

## Próximas tareas técnicas

- [ ] Añadir o corregir los módulos `models`, `detector`, `segmenter` y `websocket_manager`.
- [ ] Validar las dependencias en un entorno limpio, incluidas PyTorch, OpenCV y Ultralytics.
- [ ] Validar las imágenes recibidas y rechazar archivos no válidos.
- [ ] Definir límites de tamaño y tiempo para cargas y procesamiento.
- [ ] Restringir CORS a los orígenes necesarios.
- [ ] Generar nombres seguros para archivos temporales y garantizar su limpieza en caso de error.
- [ ] Completar la grabación y entrega del vídeo procesado.
- [ ] Gestionar desconexiones y errores de WebSocket.
- [ ] Añadir pruebas automatizadas para los endpoints y el procesamiento.
- [ ] Documentar modelos, licencias, hardware y métricas de rendimiento.

## Seguridad

La configuración actual de CORS es amplia y debe restringirse antes de publicar el servicio. El backend también necesita validación de entradas, límites de recursos y gestión segura de archivos antes de recibir datos de usuarios.

## Autor

Waly Robert

Proyecto independiente de estudio y experimentación en visión artificial aplicada.
