import os
import shutil
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from src.api.schemas import (
    UploadResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
    SearchRequest,
    SearchResponse,
    SearchChunk,
    QueryRequest,
    QueryResponse,
    SourceCitation,
    HealthResponse
)
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.database.qdrant_client import HybridQdrantClient
from src.embedding.dense_encoder import DenseEncoder
from src.embedding.sparse_encoder import SparseEncoder
from src.retrieval.hybrid_search import HybridSearchService
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Singletons / initialized services
db_client = HybridQdrantClient()
dense_encoder = DenseEncoder()
sparse_encoder = SparseEncoder()
search_service = HybridSearchService(db_client=db_client)
llm_client = LLMClient()
chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (PDF, DOCX, TXT, MD, CSV), extracts its text,
    splits it into overlapping chunks, generates dense and sparse embeddings,
    and indexes them in the hybrid Qdrant collection.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename must be provided.")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        parsed_data = DocumentParser.parse_file(file_path)
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        raise HTTPException(status_code=422, detail=f"Error parsing file: {str(e)}")

    all_chunks = []
    for page_info in parsed_data["pages"]:
        page_chunks = chunker.split_text(
            text=page_info["text"],
            metadata={
                "file_name": parsed_data["file_name"],
                "page": page_info["page"]
            }
        )
        all_chunks.extend(page_chunks)

    if not all_chunks:
        return UploadResponse(
            status="warning",
            file_name=file.filename,
            chunks_indexed=0,
            total_pages=parsed_data["total_pages"],
            file_type=parsed_data["file_type"],
            message="Document parsed but contained no extractable text chunks."
        )

    # Generate dense and sparse vectors
    chunk_texts = [c["text"] for c in all_chunks]
    try:
        dense_vectors = dense_encoder.encode_documents(chunk_texts)
        sparse_vectors = sparse_encoder.encode_documents(chunk_texts)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

    # Upsert into Qdrant
    try:
        total_upserted = db_client.upsert_chunks(
            chunks=all_chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors
        )
    except Exception as e:
        logger.error(f"Error upserting vectors to Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Database indexing failed: {str(e)}")

    return UploadResponse(
        status="success",
        file_name=file.filename,
        chunks_indexed=total_upserted,
        total_pages=parsed_data["total_pages"],
        file_type=parsed_data["file_type"],
        message=f"Successfully indexed {total_upserted} chunks across {parsed_data['total_pages']} pages."
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    Lists all documents currently indexed in the hybrid search database.
    """
    try:
        docs = db_client.list_documents()
        return DocumentListResponse(
            documents=docs,
            total_documents=len(docs)
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@router.delete("/documents/{file_name}", response_model=DeleteDocumentResponse)
async def delete_document(file_name: str):
    """
    Deletes all indexed vectors and raw files for a given document.
    """
    try:
        db_client.delete_document(file_name)
    except Exception as e:
        logger.error(f"Failed to delete document from database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete from database: {str(e)}")

    # Remove raw file from disk if present
    raw_path = os.path.join(settings.UPLOAD_DIR, file_name)
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
        except Exception as e:
            logger.warning(f"Could not remove local file {raw_path}: {e}")

    return DeleteDocumentResponse(
        status="success",
        file_name=file_name,
        message=f"Document '{file_name}' and all associated chunks were successfully deleted."
    )


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """
    Performs hybrid search (Dense + Sparse with RRF) and returns ranked chunks
    with complete score and rank transparency for both retrieval mechanisms.
    """
    try:
        results = search_service.search(
            query=request.query,
            top_k=request.top_k,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight
        )
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=[SearchChunk(**r) for r in results]
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Executes the full RAG pipeline:
    1. Retrieves relevant chunks via hybrid search (Dense + Sparse + RRF).
    2. Builds a strictly grounded citation prompt.
    3. Synthesizes a response using the active LLM provider (or grounded extractive fallback).
    4. Returns the generated answer alongside structured citations.
    """
    try:
        retrieved_chunks = search_service.search(
            query=request.query,
            top_k=request.top_k
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    prompt = PromptBuilder.build_prompt(
        query=request.query,
        context_chunks=retrieved_chunks
    )

    try:
        answer = llm_client.generate_response(prompt=prompt, context_chunks=retrieved_chunks)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    citations = [
        SourceCitation(
            file_name=c["file_name"],
            page=c["page"],
            text=c["text"],
            rrf_score=c.get("rrf_score"),
            dense_rank=c.get("dense_rank"),
            sparse_rank=c.get("sparse_rank")
        )
        for c in retrieved_chunks
    ]

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=citations,
        provider=llm_client.provider
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check for database and generation services.
    """
    try:
        qdrant_info = db_client.get_collection_info()
    except Exception as e:
        qdrant_info = {"status": "error", "error": str(e)}

    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        qdrant=qdrant_info,
        llm_provider=llm_client.provider
    )