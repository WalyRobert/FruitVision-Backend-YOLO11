# FruitVision — Backend de visão computacional

**Idioma:** Português (Brasil)  
**Status:** Protótipo experimental. A API está iniciada, mas a estrutura atual precisa de módulos e validações antes de ser considerada executável.

FruitVision é um projeto experimental de backend para análise de imagens e vídeos com visão computacional. A proposta é explorar detecção de objetos, contagem/rastreamento e segmentação, com uma API que possa atender a uma interface web.

## Objetivo

- Receber imagens para detecção.
- Receber imagens para segmentação.
- Processar vídeos.
- Disponibilizar resultados por HTTP e WebSocket.
- Evoluir para integração com uma interface React.

## Tecnologias identificadas

- Python
- FastAPI e Uvicorn
- OpenCV e NumPy
- Ultralytics / YOLO
- WebSocket
- PyTorch e bibliotecas de visão computacional listadas em `requirements.txt`

As dependências listadas representam a configuração pretendida e ainda precisam ser validadas em ambiente limpo e compatível.

## Endpoints presentes em `app.py`

| Método | Rota | Finalidade descrita |
|---|---|---|
| GET | `/health` | Verificar o estado do serviço |
| POST | `/detect` | Receber imagem para detecção |
| POST | `/segment` | Receber imagem para segmentação |
| POST | `/process-video` | Iniciar processamento de vídeo |
| WebSocket | `/ws` | Receber frames e devolver resultados de detecção |

## Módulos referenciados

O arquivo principal importa:

- `models`: modelos de resposta da API.
- `detector`: encapsulamento da detecção YOLO.
- `segmenter`: encapsulamento da segmentação.
- `websocket_manager`: gerenciamento de conexões WebSocket.

**Pendência de reprodutibilidade:** esses módulos importados não foram localizados na revisão do conteúdo atual do repositório. A aplicação pode falhar ao iniciar até que os arquivos sejam adicionados ou as importações sejam corrigidas.

## Instalação pretendida

```bash
git clone https://github.com/WalyRobert/FruitVision-Backend-YOLO11.git
cd FruitVision-Backend-YOLO11
python -m venv .venv
```

Ative o ambiente virtual e revise a compatibilidade das dependências antes de instalar:

```bash
pip install -r requirements.txt
```

O comando de inicialização pretendido é:

```bash
uvicorn app:app --reload
```

**Atenção:** a execução ainda não foi validada e pode falhar devido aos módulos ausentes ou a incompatibilidades de dependências.

## Pendências técnicas identificadas

- [ ] Adicionar ou corrigir os módulos `models`, `detector`, `segmenter` e `websocket_manager`.
- [ ] Validar dependências em ambiente limpo, incluindo PyTorch, OpenCV e Ultralytics.
- [ ] Validar imagens recebidas e rejeitar arquivos inválidos.
- [ ] Definir limites de tamanho e tempo para uploads e processamento.
- [ ] Restringir CORS às origens necessárias.
- [ ] Gerar nomes seguros para arquivos temporários e garantir limpeza em caso de falha.
- [ ] Completar gravação e disponibilização do vídeo processado.
- [ ] Tratar desconexões e erros no WebSocket.
- [ ] Adicionar testes automatizados para endpoints e processamento.
- [ ] Documentar modelos, licenças, hardware e métricas de desempenho.

## Segurança

A configuração atual de CORS é ampla e deve ser restringida antes de disponibilizar o serviço. O backend também precisa de validação de entradas, limites de recursos e gestão segura de arquivos antes de receber dados de usuários.

## Autor

Waly Robert

Projeto independente de estudo e experimentação em visão computacional aplicada.
