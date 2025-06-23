"""
Geração do índice FAISS e labels para o serviço de detecção de intenções
usando o dataset CLINC-OOS e embeddings do SentenceTransformer.
"""

import os
import json
import logging

import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from .config import MODEL_NAME, DATA_DIR

logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "all-MiniLM-L6-v2")


def build_hnsw(dim: int, m: int = 32, ef_construction: int = 400) -> faiss.Index:
    """
    Cria um índice FAISS do tipo HNSW para similaridade com inner-product (cosine).
    """
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 128
    return index


def build_index() -> tuple[str, str]:
    """
    Gera o índice FAISS e o arquivo de labels.
    Se já existirem arquivos válidos, eles são reutilizados.
    
    Returns:
        Tuple com caminhos para o arquivo .faiss e o .json de labels.
    """
    idx_path = os.path.join(DATA_DIR, f"{MODEL_NAME}.faiss")
    lbl_path = os.path.join(DATA_DIR, f"{MODEL_NAME}_labels.json")

    if os.path.exists(idx_path) and os.path.exists(lbl_path):
        try:
            faiss.read_index(idx_path)
            with open(lbl_path, "r", encoding="utf-8") as f:
                json.load(f)
            logger.info("Índice e labels já existem. Usando arquivos salvos.")
            return idx_path, lbl_path
        except (faiss.FaissException, json.JSONDecodeError, OSError) as e:
            logger.warning("Arquivos existentes inválidos. Recriando. Erro: %s", e)

    logger.info("Carregando dataset CLINC-OOS...")
    dataset = load_dataset(
        path="./datasets/clinc_oos",  # caminho local
        name="default",
        cache_dir="./datasets"
    )
    train = dataset["train"]
    texts = train["text"]
    intents = train["intent"]
    intent_names = train.features["intent"].names
    labels = [intent_names[i] for i in intents]

    logger.info("Gerando embeddings com o modelo '%s'...", MODEL_NAME)
    model = SentenceTransformer("./models/all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    array = np.array(embeddings, dtype="float32")

    if len(array.shape) != 2:
        raise ValueError("Embeddings não possuem shape 2D esperado.")

    logger.info("Construindo índice FAISS (dim=%d)...", array.shape[1])
    index = build_hnsw(array.shape[1])
    index.add(array)  # E1120 corrigido

    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("Salvando índice em '%s' e labels em '%s'...", idx_path, lbl_path)
    faiss.write_index(index, idx_path)

    with open(lbl_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"text": text, "label": label} for text, label in zip(texts, labels)],
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Geração do índice finalizada.")
    return idx_path, lbl_path
