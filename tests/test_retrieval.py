import pytest
from src.retrieval.rrf import ReciprocalRankFusion
from src.embedding.dense_encoder import DenseEncoder
from src.embedding.sparse_encoder import SparseEncoder
from src.database.qdrant_client import HybridQdrantClient
from qdrant_client import QdrantClient, models


def test_rrf_scoring_order():
    dense_results = [
        {"id": "doc1", "score": 0.95, "payload": {"text": "first in dense"}},
        {"id": "doc2", "score": 0.85, "payload": {"text": "second in dense"}},
        {"id": "doc3", "score": 0.70, "payload": {"text": "third in dense"}}
    ]
    sparse_results = [
        {"id": "doc2", "score": 12.5, "payload": {"text": "first in sparse"}},
        {"id": "doc4", "score": 10.0, "payload": {"text": "second in sparse"}},
        {"id": "doc1", "score": 8.0, "payload": {"text": "third in sparse"}}
    ]

    fused = ReciprocalRankFusion.compute_rrf(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60
    )

    # doc1 and doc2 appear in both lists, so their combined score must be higher than single-list docs
    fused_ids = [item["id"] for item in fused]
    assert "doc1" in fused_ids[:2]
    assert "doc2" in fused_ids[:2]
    assert fused[0]["rrf_score"] > fused[-1]["rrf_score"]


def test_dense_encoder_dimensions():
    encoder = DenseEncoder()
    vector = encoder.encode_text("Hybrid search with vector embeddings")
    assert isinstance(vector, list)
    assert len(vector) == 384
    for val in vector:
        assert isinstance(val, float)


def test_sparse_encoder_format():
    encoder = SparseEncoder()
    sparse_vec = encoder.encode("Hybrid search keyword matching")
    assert hasattr(sparse_vec, "indices")
    assert hasattr(sparse_vec, "values")
    assert len(sparse_vec.indices) == len(sparse_vec.values)
    assert len(sparse_vec.indices) > 0


def test_inmemory_hybrid_qdrant_client():
    client = QdrantClient(location=":memory:")
    col_name = "test_memory_hybrid"
    client.create_collection(
        collection_name=col_name,
        vectors_config={"vector_dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"vector_sparse": models.SparseVectorParams()}
    )

    point = models.PointStruct(
        id=1,
        vector={
            "vector_dense": [0.05] * 384,
            "vector_sparse": models.SparseVector(indices=[1, 2], values=[1.0, 0.5])
        },
        payload={"file_name": "test.txt", "text": "sample text", "page": 1}
    )
    client.upsert(col_name, [point])

    dense_res = client.query_points(col_name, query=[0.05] * 384, using="vector_dense", limit=1)
    assert len(dense_res.points) == 1

    sparse_res = client.query_points(
        col_name,
        query=models.SparseVector(indices=[1], values=[1.0]),
        using="vector_sparse",
        limit=1
    )
    assert len(sparse_res.points) == 1
