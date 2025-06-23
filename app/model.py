"""
Módulo de inferência: carrega modelo, índice FAISS e executa predições de intent.
"""

import os
import json
import logging
from collections import Counter

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .builder import build_index
from .config import MODEL_NAME, DATA_DIR, DEFAULT_THRESHOLD


logger = logging.getLogger(__name__)

# Configurações por ambiente
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "faiss_indices")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.7"))

# Estado global controlado internamente
_state = {
    "model": None,
    "index": None,
    "labels": None,
}


def load_model():
    """
    Carrega modelo, índice FAISS e labels.
    Se arquivos ainda não existirem, serão gerados automaticamente.
    """
    logger.info("Iniciando carregamento do modelo e índice FAISS...")

    index_path, labels_path = build_index()

    logger.info("Carregando índice FAISS de '%s'...", index_path)
    _state["index"] = faiss.read_index(index_path)

    logger.info("Carregando labels de '%s'...", labels_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        _state["labels"] = json.load(f)

    logger.info("Carregando modelo de embeddings '%s'...", MODEL_NAME)
    _state["model"] = SentenceTransformer("./models/all-MiniLM-L6-v2")

    logger.info(
        "Recursos carregados com sucesso: %d labels, índice FAISS e modelo prontos.",
        len(_state["labels"]),
    )


def predict(text: str, top_k: int = 5, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Prediz a intenção do texto de entrada.
    Retorna dicionário com query, intent predito, candidatos e scores.
    """
    if not all(_state.values()):
        raise RuntimeError("Modelo não carregado. Execute load_model() antes de usar predict().")

    logger.debug("Realizando predição para: '%s'", text)

    model = _state["model"]
    index = _state["index"]
    labels = _state["labels"]

    emb = model.encode([text], normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")
    sims, ids = index.search(emb, top_k)
    sims = sims[0].tolist()
    ids = ids[0].tolist()
    candidates = [labels[i]["label"] for i in ids]
    intent = _predict_intent(candidates, sims, threshold)

    result = {
        "query": text,
        "predicted_intent": intent,
        "candidates": candidates,
        "scores": sims,
    }
    logger.info("Resultado da predição: %s", result)
    return result


def _predict_intent(intents: list[str], sims: list[float], thr: float) -> str:
    """
    Votação majoritária entre vizinhos, com limiar para out-of-scope (OOS).
    """
    votes = Counter(intents)
    top_count = max(votes.values())
    tied = [c for c, v in votes.items() if v == top_count]

    best = max(
        tied, key=lambda c: np.mean([s for i, s in enumerate(sims) if intents[i] == c])
    )

    return "oos" if sims[0] < thr else best
