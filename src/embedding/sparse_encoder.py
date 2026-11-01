from typing import List, Dict, Any
from fastembed import SparseTextEmbedding
from qdrant_client.http import models
from config.settings import settings


class SparseEncoder:
    """
    Generates sparse token/BM25 vectors using FastEmbed.
    Default model: Qdrant/bm25.
    Outputs indices and float values directly compatible with Qdrant's sparse vector index.
    """

    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._model is None:
            self.model_name = model_name or settings.SPARSE_MODEL_NAME
            self._model = SparseTextEmbedding(model_name=self.model_name)

    def encode(self, text: str) -> models.SparseVector:
        """
        Embeds a single query string into a Qdrant SparseVector object.
        """
        if not text or not text.strip():
            return models.SparseVector(indices=[], values=[])
        res = list(self._model.embed([text]))[0]
        return models.SparseVector(
            indices=res.indices.tolist(),
            values=res.values.tolist()
        )

    def encode_dict(self, text: str) -> Dict[str, List]:
        """
        Returns raw indices and values dict for JSON serialization or logging.
        """
        if not text or not text.strip():
            return {"indices": [], "values": []}
        res = list(self._model.embed([text]))[0]
        return {
            "indices": res.indices.tolist(),
            "values": res.values.tolist()
        }

    def encode_documents(self, texts: List[str]) -> List[models.SparseVector]:
        """
        Batch embeds document chunk strings into a list of Qdrant SparseVectors.
        """
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [
            models.SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist()
            )
            for emb in embeddings
        ]
