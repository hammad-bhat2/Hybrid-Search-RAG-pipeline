from typing import List, Dict, Any, Optional


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) to merge and rerank results 
    from multiple retrieval systems (e.g., Dense semantic search and Sparse BM25).
    
    Formula:
        RRF_Score(d) = sum( weight_m / (k + rank_m(d)) ) for all systems m
    """

    @staticmethod
    def compute_rrf(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Combines dense and sparse ranked lists using weighted Reciprocal Rank Fusion.
        
        Args:
            dense_results: List of dicts with 'id', 'score', 'payload' ordered by dense similarity.
            sparse_results: List of dicts with 'id', 'score', 'payload' ordered by sparse score.
            k: Smoothing constant (default 60).
            dense_weight: Weight multiplier for dense ranks.
            sparse_weight: Weight multiplier for sparse ranks.
            
        Returns:
            List of merged document items sorted descending by RRF score.
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Results (1-indexed rank)
        for rank, item in enumerate(dense_results, start=1):
            doc_id = str(item["id"])
            item_score = float(item.get("score", 0.0))
            dense_rrf = dense_weight / (k + rank)

            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "payload": item.get("payload", {}),
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "dense_score": item_score,
                    "sparse_rank": None,
                    "sparse_score": 0.0
                }
            scores[doc_id]["rrf_score"] += dense_rrf

        # 2. Process Sparse Results (1-indexed rank)
        for rank, item in enumerate(sparse_results, start=1):
            doc_id = str(item["id"])
            item_score = float(item.get("score", 0.0))
            sparse_rrf = sparse_weight / (k + rank)

            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "payload": item.get("payload", {}),
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "sparse_rank": rank,
                    "sparse_score": item_score
                }
            else:
                scores[doc_id]["sparse_rank"] = rank
                scores[doc_id]["sparse_score"] = item_score

            scores[doc_id]["rrf_score"] += sparse_rrf

        # 3. Sort by fused RRF score descending
        fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return fused
