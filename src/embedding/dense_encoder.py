from typing import List
from fastembed import TextEmbedding
from config.settings import settings


class DenseEncoder:
    """
    Generates dense vector embeddings using FastEmbed (ONNX runtime).
    Default model: BAAI/bge-small-en-v1.5 (dimension: 384).
    """

    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._model is None:
            self.model_name = model_name or settings.DENSE_MODEL_NAME
            self._model = TextEmbedding(model_name=self.model_name)

    def encode_text(self, text: str) -> List[float]:
        """
        Embeds a single query string into a dense vector (float list).
        """
        if not text or not text.strip():
            return [0.0] * settings.DENSE_VECTOR_SIZE
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds document chunk strings into a list of dense vectors.
        """
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]
