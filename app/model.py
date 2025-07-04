"""
Módulo de inferência: encapsula o modelo de detecção de intenção e sua lógica.
"""

import os
import json
import logging
from collections import Counter

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .builder import build_index
from .config import DATA_DIR, DEFAULT_THRESHOLD, MODEL_NAME

logger = logging.getLogger(__name__)


class IntentModel:
    """
    Encapsula o modelo SentenceTransformer, o índice FAISS e os labels
    para realizar a predição de intenções.
    """

    def __init__(self, model_name: str = MODEL_NAME, data_dir: str = DATA_DIR):
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.index = None
        self.labels = None

    def load(self):
        """
        Carrega o modelo, o índice FAISS e os labels.
        Se os artefatos não existirem, eles são gerados automaticamente.
        """
        logger.info("Iniciando carregamento do modelo e do índice FAISS...")

        index_path, labels_path = build_index()

        logger.info("Carregando índice FAISS de '%s'...", index_path)
        self.index = faiss.read_index(index_path)

        logger.info("Carregando labels de '%s'...", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        logger.info("Carregando modelo de embeddings '%s'...", self.model_name)
        local_model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", self.model_name
        )
        self.model = SentenceTransformer(local_model_path, trust_remote_code=True)

        logger.info(
            "Recursos carregados com sucesso: %d labels, índice FAISS e modelo prontos.",
            len(self.labels),
        )

    def predict(self, text: str, top_k: int, threshold: float) -> dict:
        """
        Prediz a intenção do texto de entrada.
        Retorna um dicionário com a query, a intenção prevista, os candidatos e os scores.
        """
        if not all((self.model, self.index, self.labels)):
            raise RuntimeError(
                "Modelo não carregado. Execute o método load() antes de usar predict()."
            )

        logger.debug("Realizando predição para: '%s'", text)

        emb = self.model.encode([text], normalize_embeddings=True)
        emb = np.array(emb, dtype="float32")
        sims, ids = self.index.search(emb, top_k)

        sims = sims[0].tolist()
        ids = ids[0].tolist()
        candidates = [self.labels[i]["label"] for i in ids]
        intent = self._predict_intent(candidates, sims, threshold)

        result = {
            "query": text,
            "predicted_intent": intent,
            "candidates": candidates,
            "scores": sims,
        }
        logger.info("Resultado da predição: %s", result)
        return result

    @staticmethod
    def _predict_intent(intents: list[str], sims: list[float], thr: float) -> str:
        """
        Realiza uma votação majoritária entre os vizinhos mais próximos,
        com um limiar para classificar como out-of-scope (OOS).
        """
        if not sims or sims[0] < thr:
            return "oos"

        votes = Counter(intents)
        top_count = max(votes.values())
        tied = [c for c, v in votes.items() if v == top_count]

        # Em caso de empate, desempata pela média de similaridade
        best = max(
            tied, key=lambda c: np.mean([s for i, s in enumerate(sims) if intents[i] == c])
        )

        return best


# --- Interface Pública do Módulo ---

# Instância única (singleton) do modelo para ser usada pela aplicação.
_intent_model = IntentModel()


def load_model():
    """Carrega o modelo singleton, o índice e os labels."""
    _intent_model.load()


def predict(text: str, top_k: int = 5, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Executa a predição usando a instância singleton do modelo."""
    return _intent_model.predict(text, top_k=top_k, threshold=threshold)
