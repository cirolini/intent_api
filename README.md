# Detecção de Intenções via Recuperação Vetorial

Este pacote fornece uma API REST simples em Flask para detecção de intenções (intents) em texto,
utilizando embeddings de sentenças e busca em índice FAISS.

## Estrutura do Diretório

```
intent_api/
├── app/
│   ├── __init__.py     # Configuração do Flask e logging
│   ├── api.py          # Endpoints REST (health, predict)
│   ├── model.py        # Carregamento de modelo, índice FAISS e lógica de inferência
│   └── utils.py        # Funções de pré-processamento de texto
├── requirements.txt    # Dependências Python
└── run.py              # Script de inicialização da aplicação
```

## Pré-requisitos

- Python 3.8+
- `virtualenv` ou suporte a `venv`

## Instalação

```bash
# A partir da pasta intent_api/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Variáveis de Ambiente

- `MODEL_NAME`    Nome do modelo Hugging Face (padrão: `all-MiniLM-L6-v2`).
- `DATA_DIR`      Diretório onde estão os arquivos do índice FAISS e labels (padrão: `faiss_indices`).
- `THRESHOLD`     Limiar de similaridade para classificação como out-of-scope (padrão: `0.7`).
- `LOG_LEVEL`     Nível de log do Flask (padrão: `INFO`).
- `HOST`, `PORT`, `DEBUG`  Configurações de host, porta e modo debug da API.

## Construção do Índice FAISS

O notebook na raiz do projeto (`efficient-intent-detection-via-vector-retrieval.ipynb`)
contém o fluxo para gerar o índice FAISS e o arquivo de labels:

- embeddings das frases de treinamento
- índice FAISS (`<MODEL_NAME>.faiss`)
- arquivo de labels (`<MODEL_NAME>_labels.json`)

Execute-o para gerar a pasta `faiss_indices/` (ou ajuste `DATA_DIR`).

## Como Executar

```bash
# Com o ambiente virtual ativado e índice gerado
python run.py
```

A API estará disponível em `http://<HOST>:<PORT>` (padrão `0.0.0.0:5000`).

### Endpoints

- `GET /health`     Verifica o status do serviço.
- `POST /predict`   Prediz intenções para um texto.
- `POST /build_index`   Gera ou carrega o índice FAISS e o arquivo de labels a partir do dataset CLINC-OOS.

#### Exemplo de Requisição `/predict`

```bash
curl -X POST http://localhost:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"text": "Olá, gostaria de abrir uma conta", "top_k": 5, "threshold": 0.75}'
```

#### Exemplo de Resposta

```json
{
  "query": "olá gostaria de abrir uma conta",
  "predicted_intent": "open_account",
  "candidates": ["open_account","balance_inquiry",...],
  "scores": [0.92, 0.85, ...]
}
```

## Contribuição

Pull requests são bem-vindos! Para alterações maiores, abra uma issue primeiro para discutirmos o escopo.