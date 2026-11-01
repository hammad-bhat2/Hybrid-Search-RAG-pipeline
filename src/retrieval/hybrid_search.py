import logging
from typing import List, Dict, Any, Optional
from src.database.qdrant_client import HybridQdrantClient
from src.embedding.dense_encoder import DenseEncoder
from src.embedding.sparse_encoder import SparseEncoder
from src.retrieval.rrf import ReciprocalRankFusion
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Orchestrates hybrid retrieval by querying both dense and sparse vector spaces
    and fusing their ranked outputs using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, db_client: Optional[HybridQdrantClient] = None):
        self.db_client = db_client or HybridQdrantClient()
        self.dense_encoder = DenseEncoder()
        self.sparse_encoder = SparseEncoder()

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search and returns top_k fused results.
        """
        if not query or not query.strip():
            return []

        dw = dense_weight if dense_weight is not None else settings.DENSE_WEIGHT
        sw = sparse_weight if sparse_weight is not None else settings.SPARSE_WEIGHT

        # 1. Encode query in both vector representations
        dense_vector = self.dense_encoder.encode_text(query)
        sparse_vector = self.sparse_encoder.encode(query)

        # Retrieve a broader pool (3x top_k) for effective rank fusion
        candidate_limit = max(top_k * 3, 15)

        dense_points = []
        sparse_points = []

        try:
            dense_points = self.db_client.query_dense(dense_vector, limit=candidate_limit)
        except Exception as e:
            logger.error(f"Error during dense vector search: {e}")

        try:
            sparse_points = self.db_client.query_sparse(sparse_vector, limit=candidate_limit)
        except Exception as e:
            logger.error(f"Error during sparse vector search: {e}")

        # Format candidates for RRF
        dense_results = [
            {"id": str(p.id), "score": float(p.score), "payload": p.payload}
            for p in dense_points
        ]
        sparse_results = [
            {"id": str(p.id), "score": float(p.score), "payload": p.payload}
            for p in sparse_points
        ]

        # 2. Fuse rankings via RRF
        fused = ReciprocalRankFusion.compute_rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=settings.RRF_K,
            dense_weight=dw,
            sparse_weight=sw
        )

        # 3. Format and return top_k results
        top_results = []
        for item in fused[:top_k]:
            payload = item.get("payload", {})
            top_results.append({
                "id": item["id"],
                "text": payload.get("text", ""),
                "file_name": payload.get("file_name", "unknown"),
                "page": payload.get("page", 1),
                "chunk_index": payload.get("chunk_index", 0),
                "rrf_score": round(item["rrf_score"], 5),
                "dense_rank": item["dense_rank"],
                "dense_score": round(item["dense_score"], 4),
                "sparse_rank": item["sparse_rank"],
                "sparse_score": round(item["sparse_score"], 4)
            })

        return top_results