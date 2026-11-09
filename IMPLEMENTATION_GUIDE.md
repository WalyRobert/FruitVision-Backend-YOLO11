# Guía de integración / Guia de integração

Consulte [README.md](README.md) para instalar e iniciar a API. Esta versão executável substitui o esqueleto que importava módulos ausentes.

## Contrato para o frontend

- `GET /health`: `status`, `mode` (`demo` ou `yolo`), `inference_enabled`, modelos carregados e `authentication_required`.
- `POST /detect`: multipart com campo `file`. Retorna `success`, `mode`, `detections`, `image_shape` como `[largura, altura]` e `processing_time_ms`. Cada detecção contém `class_id`, `label`, `confidence` entre 0 e 1 e `bbox` como `[x1,y1,x2,y2]` em pixels.
- `POST /segment`: multipart `file`; retorna polígonos de segmentação 2D, quantidade, modo e tempo.
- `POST /process-video`: multipart `file` e `enable_segmentation`. Retorna metadados, soma de detecções por frame e `download_url` no formato `/videos/<id>`.
- `GET /videos/<id>`: MP4 temporário, autenticado quando há chave configurada; expira após uma hora.
- `/ws`: protocolo documentado no README. A interface pode usar HTTP sem implementar WebSocket.

No modo demo, imagens e vídeos são validados/processados, mas as listas de detecção e segmentação ficam vazias. O frontend deve mostrar esse modo e não inventar confiança, produtividade, qualidade ou contagens únicas.

Quando `FRUITVISION_API_KEY` está configurada, envie `X-API-Key` pelo servidor da interface. Evite embutir chaves em bundles ou URLs. `ALLOWED_ORIGINS` define CORS; `FRUITVISION_CORS_ORIGINS` é aceito como alternativa compatível com a proposta anterior de hardening. O servidor padrão escuta em loopback.

Pesos reais não estão incluídos nem foram usados nos testes. As dependências de ML são opcionais e exigem configuração e licenciamento apropriados. Não há implementação SAM 3D ou ObjectCounter nesta edição.
