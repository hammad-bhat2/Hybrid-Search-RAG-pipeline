import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridQdrantClient:
    """
    Manages connection and operations on Qdrant vector database for hybrid search.
    Supports both client-server network mode and embedded disk storage mode.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None
    ):
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.client = self._init_client()
        self._ensure_collection_exists()

    def _init_client(self) -> QdrantClient:
        # Check if remote URL is configured
        if settings.QDRANT_URL:
            logger.info(f"Connecting to Qdrant URL: {settings.QDRANT_URL}")
            return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

        # Attempt network connection to Qdrant host/port
        try:
            client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=2.0,
                api_key=settings.QDRANT_API_KEY
            )
            # Ping to verify server is live
            client.get_collections()
            logger.info(f"Connected to Qdrant server at {self.host}:{self.port}")
            return client
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant server at {self.host}:{self.port} ({e}). "
                f"Falling back to local embedded storage at {settings.QDRANT_STORAGE_PATH}"
            )
            os.makedirs(settings.QDRANT_STORAGE_PATH, exist_ok=True)
            return QdrantClient(path=settings.QDRANT_STORAGE_PATH)

    def _ensure_collection_exists(self):
        """
        Ensures the hybrid collection exists with dense and sparse vector configurations.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                logger.info(f"Creating hybrid collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "vector_dense": VectorParams(
                            size=settings.DENSE_VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "vector_sparse": SparseVectorParams()
                    }
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        dense_vectors: List[List[float]],
        sparse_vectors: List[models.SparseVector]
    ) -> int:
        """
        Upserts chunk payloads with dense and sparse vectors into Qdrant.
        """
        if not chunks:
            return 0

        points = []
        for i, chunk in enumerate(chunks):
            point = models.PointStruct(
                id=chunk["id"],
                vector={
                    "vector_dense": dense_vectors[i],
                    "vector_sparse": sparse_vectors[i]
                },
                payload={
                    "text": chunk["text"],
                    "file_name": chunk["metadata"]["file_name"],
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "word_count": chunk["metadata"].get("word_count", 0),
                    "char_count": chunk["metadata"].get("char_count", 0)
                }
            )
            points.append(point)

        # Batch in groups of 100
        batch_size = 100
        total_upserted = 0
        for b_start in range(0, len(points), batch_size):
            batch = points[b_start:b_start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)

        return total_upserted

    def query_dense(self, query_vector: List[float], limit: int = 10) -> List[Any]:
        """
        Performs vector similarity search on the dense vector space.
        """
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="vector_dense",
            limit=limit,
            with_payload=True
        )
        return res.points

    def query_sparse(self, sparse_vector: models.SparseVector, limit: int = 10) -> List[Any]:
        """
        Performs keyword/BM25 search on the sparse vector space.
        """
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="vector_sparse",
            limit=limit,
            with_payload=True
        )
        return res.points

    def delete_document(self, file_name: str) -> bool:
        """
        Deletes all chunks/points belonging to a given file name.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_name",
                        match=models.MatchValue(value=file_name)
                    )
                ]
            ),
            wait=True
        )
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieves summary of all indexed documents in the collection.
        """
        docs_map: Dict[str, Dict[str, Any]] = {}
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for rec in records:
                payload = rec.payload or {}
                file_name = payload.get("file_name", "unknown")
                page = payload.get("page", 1)

                if file_name not in docs_map:
                    docs_map[file_name] = {
                        "file_name": file_name,
                        "total_chunks": 0,
                        "pages": set(),
                        "total_words": 0
                    }

                docs_map[file_name]["total_chunks"] += 1
                docs_map[file_name]["pages"].add(page)
                docs_map[file_name]["total_words"] += payload.get("word_count", 0)

            if offset is None:
                break

        result = []
        for file_name, data in docs_map.items():
            result.append({
                "file_name": file_name,
                "total_chunks": data["total_chunks"],
                "total_pages": len(data["pages"]),
                "total_words": data["total_words"]
            })

        return sorted(result, key=lambda x: x["file_name"])

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Returns status information about the active collection.
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "collection_name": self.collection_name,
            "status": info.status,
            "points_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
        }