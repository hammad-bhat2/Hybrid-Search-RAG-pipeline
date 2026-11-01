from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    status: str
    file_name: str
    chunks_indexed: int
    total_pages: int
    file_type: str
    message: str


class DocumentInfo(BaseModel):
    file_name: str
    total_chunks: int
    total_pages: int
    total_words: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int


class DeleteDocumentResponse(BaseModel):
    status: str
    file_name: str
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    dense_weight: Optional[float] = Field(1.0, ge=0.0, le=10.0, description="Weight multiplier for dense ranks")
    sparse_weight: Optional[float] = Field(1.0, ge=0.0, le=10.0, description="Weight multiplier for sparse ranks")


class SearchChunk(BaseModel):
    id: str
    text: str
    file_name: str
    page: int
    chunk_index: int
    rrf_score: float
    dense_rank: Optional[int]
    dense_score: Optional[float]
    sparse_rank: Optional[int]
    sparse_score: Optional[float]


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchChunk]


class SourceCitation(BaseModel):
    file_name: str
    page: Optional[int] = 1
    text: str
    rrf_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question for the RAG pipeline")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of retrieved chunks for context")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    provider: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    qdrant: Dict[str, Any]
    llm_provider: str