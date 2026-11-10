# FruitVision — API de visão computacional

API FastAPI para imagens, vídeos curtos e frames por WebSocket. A estrutura agora inclui os módulos importados, validação de entrada, gravação de vídeos e testes. **Modo padrão: demo, sem inferência de IA.** Nesse modo os resultados de detecção e segmentação são vazios e a resposta informa `mode=demo`.

## Instalar e iniciar

Python 3.12 recomendado. Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe app.py
```

Abra **http://127.0.0.1:8000/docs** para a interface da API e **/health** para o estado do serviço. Linux/macOS: substitua `.venv\Scripts\python.exe` por `.venv/bin/python`.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Ativar YOLO real

1. Instale a distribuição PyTorch adequada a CPU/GPU pelo [guia oficial](https://pytorch.org/get-started/locally/).
2. Instale `requirements-ml.txt`. Ultralytics requer OpenCV padrão; use somente uma variante OpenCV por ambiente. Se necessário, remova `opencv-python-headless` após instalar a variante padrão.
3. Obtenha pesos confiáveis compatíveis com YOLO11 e salve-os localmente; `yolo11n.pt` para detecção, `yolo11n-seg.pt` para segmentação 2D. O servidor não baixa pesos implicitamente.
4. Configure variáveis e inicie:

```powershell
$env:FRUITVISION_MODE="yolo"
$env:YOLO_MODEL="C:\modelos\yolo11n.pt"
$env:YOLO_SEGMENT_MODEL="C:\modelos\yolo11n-seg.pt"
.venv\Scripts\python.exe app.py
```

O carregamento é sob demanda. Pesos ausentes resultam em HTTP 503; `/health` distingue modo e modelos carregados. Os testes desta entrega não carregaram pesos reais nem mediram precisão, FPS ou GPU. A configuração `.env.example` é uma referência: exporte as variáveis no terminal; o arquivo `.env` não é carregado automaticamente.

As classes padrão são `apple,banana,orange`. Configure `FRUIT_CLASSES` com os nomes das classes do seu modelo. Detectar outras frutas exige pesos treinados para essas classes. Modelos COCO não equivalem a um detector validado para toda produção agrícola.

## Rotas

| Rota | Uso |
| --- | --- |
| `GET /health` | Modo, autenticação exigida e estado dos modelos |
| `POST /detect` | Multipart `file`; caixas, classes e confiança |
| `POST /segment` | Multipart `file`; polígonos de segmentação 2D |
| `POST /process-video` | Multipart `file` e `enable_segmentation=true/false`; metadados e URL de download |
| `GET /videos/{id}` | Download do MP4 gerado, disponível por até 1 hora |
| `WS /ws` | Frames hex ou base64; resultados por frame |

Exemplo com curl:

```bash
curl -F "file=@foto.png" http://127.0.0.1:8000/detect
curl -F "file=@trecho.mp4" -F "enable_segmentation=true" http://127.0.0.1:8000/process-video
```

O vídeo é gravado frame a frame, com caixas e polígonos quando disponíveis. O contêiner é MP4 com codec `mp4v`; reprodução em navegadores depende dos codecs disponíveis e pode exigir transcodificação. Áudio não é preservado. `total_detections` soma detecções por frame; **não representa frutas únicas rastreadas ou contagem por cruzamento de linha**. Não há implementação SAM 3D, reconstrução 3D ou rastreamento ObjectCounter nesta versão. A segmentação disponível usa YOLO 2D e é identificada na resposta.

## Limites e operação

- Upload padrão: 25 MB, limitado também antes do armazenamento multipart.
- Imagem/frame: até 12 milhões de pixels; imagens são verificadas antes da decodificação OpenCV.
- Vídeo: até 300 frames, 30 segundos de duração e verificação do tempo entre frames. O prazo não interrompe uma inferência individual em andamento.
- Um processamento por instância; concorrência retorna HTTP 429. O trabalho pesado roda em thread, preservando o atendimento de `/health`.
- Nomes de arquivos são UUIDs, independentes do nome recebido. Temporários e saídas parciais são removidos nas falhas.
- Saídas ficam em `data/outputs/`, com limite de 20 arquivos. A remoção de arquivos expirados ocorre ao iniciar novo processamento de vídeo.
- CORS usa lista explícita em `ALLOWED_ORIGINS`. Defina `FRUITVISION_API_KEY` e envie `X-API-Key` para as rotas de processamento/download quando compartilhar o serviço.
- O servidor padrão escuta apenas `127.0.0.1`. A API é local; exposição externa exige proxy com limite de corpo, autenticação e controle de recursos. Para volumes maiores, substitua o processamento síncrono da requisição por uma fila de tarefas.

## WebSocket

Envie `{"type":"frame","frame":"<hex dos bytes da imagem>"}` ou acrescente `"encoding":"base64"` para base64. O servidor responde primeiro `type=ready`, depois `type=detection_results` ou `type=error`. O limite de mensagem é 2 MB e o timeout entre mensagens é 30 segundos.

Quando há chave, clientes que não conseguem definir headers devem enviar primeiro `{"api_key":"sua-chave"}`. Não coloque a chave na URL. Para navegador, a origem também deve estar em `ALLOWED_ORIGINS`.

## Evidências de teste

Os testes usam detectores controlados para verificar os contratos da API; o modo demo retorna listas vazias. Há testes de imagem inválida, upload excessivo, origem/credencial, WebSocket, concorrência, falta de pesos e vídeo. O teste de vídeo cria um AVI curto, processa, baixa o MP4 e decodifica seus frames. Isso comprova a tubulação de mídia, não a qualidade dos modelos de IA.

Referências: [detecção Ultralytics](https://docs.ultralytics.com/tasks/detect/), [segmentação Ultralytics](https://docs.ultralytics.com/tasks/segment/).

Autor: Waly Robert.

Para reproduzir exatamente as dependências de teste desta entrega, instale requirements-test-lock.txt. O CI usa esse arquivo; a instalação opcional de ML não foi incluída nesse lock.

## Diagnóstico local

Execute python diagnostico.py no ambiente virtual para conferir dependências e presença dos pesos exigidos pelo modo YOLO, sem baixar modelos, consumir serviços externos ou mostrar credenciais. Presença do arquivo não comprova que o modelo carrega nem sua precisão.

