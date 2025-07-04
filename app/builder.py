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


class IndexBuilder:
    """
    Encapsula a lógica de construção do índice FAISS e dos labels
    a partir do dataset CLINC-OOS.
    """

    def __init__(self, model_name: str = MODEL_NAME, data_dir: str = DATA_DIR):
        """
        Inicializa o builder.

        Args:
            model_name (str): Nome do modelo a ser usado.
            data_dir (str): Diretório para salvar/carregar o índice e labels.
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.model_path = LOCAL_MODEL_PATH
        self.index_path = os.path.join(self.data_dir, f"{self.model_name}.faiss")
        self.labels_path = os.path.join(self.data_dir, f"{self.model_name}_labels.json")
        self.model = None

    def _load_model(self):
        """Carrega o modelo SentenceTransformer."""
        logger.info("Carregando modelo de embeddings '%s'...", self.model_name)
        self.model = SentenceTransformer(self.model_path, trust_remote_code=True)

    def _load_dataset(self) -> tuple[list[str], list[str]]:
        """Carrega o dataset CLINC-OOS e extrai textos e labels."""
        logger.info("Carregando dataset CLINC-OOS...")
        dataset = load_dataset(
            path="./datasets/clinc_oos", name="default", cache_dir="./datasets"
        )
        train = dataset["train"]
        texts = train["text"]
        intents = train["intent"]
        intent_names = train.features["intent"].names
        labels = [intent_names[i] for i in intents]
        return texts, labels

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Gera embeddings para uma lista de textos."""
        if not self.model:
            self._load_model()

        logger.info("Gerando embeddings com o modelo '%s'...", self.model_name)
        embeddings = self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        array = np.array(embeddings, dtype="float32")
        if len(array.shape) != 2:
            raise ValueError("Embeddings não possuem shape 2D esperado.")
        return array

    @staticmethod
    def _build_hnsw_index(dim: int, m: int = 32, ef_construction: int = 400) -> faiss.Index:
        """Cria um índice FAISS do tipo HNSW para similaridade com inner-product (cosine)."""
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = 128
        return index

    def _create_and_save_artifacts(self) -> tuple[str, str]:
        """Orquestra a criação e salvamento do índice e labels."""
        texts, labels = self._load_dataset()
        embeddings = self._generate_embeddings(texts)

        logger.info("Construindo índice FAISS (dim=%d)...", embeddings.shape[1])
        index = self._build_hnsw_index(embeddings.shape[1])
        index.add(embeddings)

        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("Salvando índice em '%s' e labels em '%s'...", self.index_path, self.labels_path)
        faiss.write_index(index, self.index_path)

        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"text": text, "label": label} for text, label in zip(texts, labels)],
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Geração do índice finalizada.")
        return self.index_path, self.labels_path

    def _artifacts_are_valid(self) -> bool:
        """Verifica se o índice e os labels existem e são válidos."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.labels_path):
            return False
        try:
            faiss.read_index(self.index_path)
            with open(self.labels_path, "r", encoding="utf-8") as f:
                json.load(f)
            return True
        except (faiss.FaissException, json.JSONDecodeError, OSError) as e:
            logger.warning("Erro ao validar artefatos existentes: %s", e)
            return False

    def build(self) -> tuple[str, str]:
        """
        Gera o índice FAISS e o arquivo de labels.
        Se já existirem arquivos válidos, eles são reutilizados.
        
        Returns:
            Tuple com caminhos para o arquivo .faiss e o .json de labels.
        """
        if self._artifacts_are_valid():
            logger.info("Índice e labels já existem. Usando arquivos salvos.")
            return self.index_path, self.labels_path
        
        logger.warning("Artefatos não encontrados ou inválidos. Recriando...")
        return self._create_and_save_artifacts()


def build_index() -> tuple[str, str]:
    """
    Interface pública para construir o índice. Instancia e executa o IndexBuilder.
    """
    builder = IndexBuilder()
    return builder.build()
