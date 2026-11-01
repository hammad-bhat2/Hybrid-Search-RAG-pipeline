import pytest
import io
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "api_prefix" in data

    # Root endpoint serves frontend HTML or online status
    html_res = client.get("/")
    assert html_res.status_code == 200


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "qdrant" in data


def test_list_documents_initially():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_upload_and_query_flow():
    # 1. Upload a test text file
    file_content = b"Qdrant is an open-source vector search engine. It supports hybrid search with dense and sparse vectors via reciprocal rank fusion."
    test_file = io.BytesIO(file_content)

    response = client.post(
        "/api/v1/upload",
        files={"file": ("test_doc.txt", test_file, "text/plain")}
    )
    assert response.status_code == 201
    upload_data = response.json()
    assert upload_data["status"] == "success"
    assert upload_data["chunks_indexed"] >= 1
    assert upload_data["file_name"] == "test_doc.txt"

    # 2. Verify document is listed
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    doc_names = [d["file_name"] for d in list_res.json()["documents"]]
    assert "test_doc.txt" in doc_names

    # 3. Perform hybrid search
    search_res = client.post(
        "/api/v1/search",
        json={"query": "What is Qdrant?", "top_k": 3}
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_results"] >= 1
    first_result = search_data["results"][0]
    assert "vector search engine" in first_result["text"]
    assert first_result["rrf_score"] > 0

    # 4. Perform full RAG query
    query_res = client.post(
        "/api/v1/query",
        json={"query": "What does Qdrant support?", "top_k": 3}
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert "answer" in query_data
    assert len(query_data["sources"]) >= 1
    assert query_data["sources"][0]["file_name"] == "test_doc.txt"

    # 5. Clean up / delete document
    del_res = client.delete("/api/v1/documents/test_doc.txt")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"
