# Detecção de Intenções via Recuperação Vetorial

Este projeto fornece uma API REST simples em Flask para detecção de intenções (intents) em textos, utilizando embeddings de sentenças e busca vetorial com FAISS.

## Estrutura do Projeto

```
intent_api/
├── app/
│   ├── __init__.py     # Configuração do Flask e logging
│   ├── api.py          # Endpoints REST (health, predict, build_index)
│   ├── model.py        # Carregamento de modelo, FAISS e lógica de inferência
│   ├── builder.py      # Geração e serialização do índice FAISS
│   ├── config.py       # Leitura de variáveis de ambiente
│   └── utils.py        # Funções auxiliares e pré-processamento de texto
├── requirements.txt    # Dependências Python
├── run.py              # Inicialização da aplicação
└── Dockerfile          # Containerização da API (opcional)
```

## Pré-requisitos

- Python 3.8 ou superior
- `virtualenv` ou suporte a `venv` nativo

## Instalação

```bash
# Clone o repositório e acesse a pasta
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Variáveis de Ambiente

Configure via `.env` ou exportando diretamente:

- `MODEL_NAME`: nome do modelo Hugging Face (padrão: `all-MiniLM-L6-v2`)
- `DATA_DIR`: diretório onde ficam os arquivos FAISS e labels (padrão: `faiss_indices`)
- `THRESHOLD`: limiar de similaridade para rejeitar intents fora do escopo (default: `0.7`)
- `LOG_LEVEL`: nível de log do Flask (default: `INFO`)
- `HOST`, `PORT`, `DEBUG`: configurações do servidor

## Construção do Índice FAISS

Para gerar o índice FAISS e o arquivo de labels, utilize o notebook `efficient-intent-detection-via-vector-retrieval.ipynb` (não incluído neste pacote).

Esse processo cria os seguintes arquivos dentro da pasta `faiss_indices/`:

- `<MODEL_NAME>.faiss`: índice vetorial
- `<MODEL_NAME>_labels.json`: labels associadas aos vetores

Caso use outro diretório, ajuste a variável `DATA_DIR`.

## Como Executar

Após instalar as dependências e gerar o índice FAISS:

```bash
python run.py
```

A API será iniciada em `http://0.0.0.0:5000` por padrão.

## Endpoints da API

- `GET /health`: verifica se a API está no ar
- `POST /predict`: retorna as intenções mais próximas para um texto
- `POST /build_index`: gera/recarrega o índice a partir do dataset CLINC-OOS

### Exemplo de Requisição `/predict`

```bash
curl -X POST http://localhost:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"text": "Olá, gostaria de abrir uma conta", "top_k": 5, "threshold": 0.75}'
```

### Exemplo de Resposta

```json
{
  "query": "olá gostaria de abrir uma conta",
  "predicted_intent": "open_account",
  "candidates": ["open_account", "balance_inquiry", ...],
  "scores": [0.92, 0.85, ...]
}
```

## Contribuições

Pull requests são bem-vindos! Para mudanças maiores, por favor abra uma issue para discutirmos primeiro.
