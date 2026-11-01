# ⚡ Production-Ready Hybrid Search RAG Pipeline

A high-performance, production-ready **Retrieval-Augmented Generation (RAG)** pipeline combining **Dense Semantic Search** (FastEmbed `BAAI/bge-small-en-v1.5`), **Sparse Keyword Matching** (FastEmbed `Qdrant/bm25`), **Reciprocal Rank Fusion (RRF)**, and strict-grounding LLM synthesis with source citations.

---

## 🌟 Key Features

- **Hybrid Retrieval Architecture**: Blends deep semantic vector search with exact lexical BM25 keyword matching to eliminate hallucinations and capture precise domain keywords.
- **Reciprocal Rank Fusion (RRF)**: Merges ranked results via $RRF(d) = \sum \frac{w_m}{k + \text{rank}_m(d)}$ with customizable weights and rank transparency.
- **Multi-Format Ingestion**: Ingests and extracts structured text from **PDF**, **DOCX**, **TXT**, **Markdown**, and **CSV** files with page-level tracking.
- **Smart Chunking**: Overlapping sliding-window chunker preserving semantic boundaries, word counts, and deterministic point hashes.
- **Resilient Vector Storage**: Seamless dual-mode Qdrant integration:
  - Runs in **local embedded storage** (`./data/qdrant_db`) out-of-the-box with **zero external dependencies**.
  - Automatically connects to **Qdrant Server / Docker Cluster** when available.
- **Multi-Provider Generation**:
  - OpenAI (`gpt-4o-mini`, `gpt-4o`)
  - Google Gemini (`gemini-2.5-flash`)
  - **Intelligent Local Extractive Fallback**: Operates completely offline without external API keys for local dev and CI/CD testing.
- **Interactive Web UI**: Modern dark-slate React 18 + Vite frontend with:
  - Drag-and-drop document upload and indexed documents management.
  - Chat interface with inline citation badges and expandable context cards.
  - Live **Hybrid Search Inspector** displaying Dense vs. Sparse vs. Fused RRF rankings.
- **Production Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` orchestrating Qdrant and the API service.

---

## 📐 Architecture

```
                       +-----------------------------+
                       |       Raw Documents         |
                       | (PDF, DOCX, TXT, MD, CSV)   |
                       +--------------+--------------+
                                      |
                              DocumentParser
                                      |
                               DocumentChunker
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
            DenseEncoder                            SparseEncoder
     (BAAI/bge-small-en-v1.5)                       (Qdrant/BM25)
           (384-dim)                             (Indices & Values)
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                        +---------------------------+
                        |     Hybrid Qdrant DB      |
                        | (Local Embedded / Server) |
                        +-------------+-------------+
                                      |
       User Query --------------------+
            |                         |
            v                         v
      Query Embeddings        Vector Retrieval
      (Dense + Sparse)       (Dense & Sparse Top-K)
            |                         |
            +-------------------------+
                                      |
                                      v
                        +---------------------------+
                        |   Reciprocal Rank Fusion  |
                        |       (RRF k=60)          |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |   Grounded PromptBuilder  |
                        |     + LLMClient           |
                        |  (OpenAI / Gemini /       |
                        |      Offline Fallback)    |
                        +-------------+-------------+
                                      |
                                      v
                          Answer + Source Citations
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11)
- **Node.js 18+** (for frontend build)

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Frontend UI
```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Configure Environment (Optional)
Copy the template configuration:
```bash
cp .env.example .env
```
*(Optional)* Add your `OPENAI_API_KEY` or `GEMINI_API_KEY`. If omitted, the pipeline runs seamlessly using its built-in local extractive generator!

### 5. Start Application
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Web App**: [http://localhost:8000](http://localhost:8000)
- **Swagger / OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🐳 Docker Deployment

To launch the full production stack (Qdrant Vector Database + FastAPI Backend + Static UI):

```bash
docker-compose up --build -d
```

Check service status:
```bash
docker-compose ps
docker-compose logs -f backend
```

---

## 📡 REST API Reference

### 1. Ingest Document
Uploads a document, parses pages, chunks content, generates dual embeddings, and indexes points into Qdrant.
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_report.pdf"
```

### 2. List Indexed Documents
```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

### 3. Delete Document
Deletes all indexed vectors and raw storage associated with the document.
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/sample_report.pdf"
```

### 4. Hybrid Search (Rank Transparency)
Executes dense vector + sparse BM25 search and returns results fused by RRF with full ranking breakdown.
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latency benchmarks?",
    "top_k": 5,
    "dense_weight": 1.0,
    "sparse_weight": 1.0
  }'
```

### 5. Grounded RAG Query
Executes hybrid retrieval and generates an answer strictly grounded in retrieved passages with source citations.
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is reciprocal rank fusion calculated?",
    "top_k": 5
  }'
```

### 6. Health Check
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

---

## 🧪 Running Automated Tests

Run the test suite with verbose output:

```bash
python -m pytest tests/ -v
```

All 16 unit and integration tests verify:
- Document parsing (PDF, DOCX, TXT, CSV)
- Sliding-window chunker with boundary preservation
- Dense vector dimension (384) & sparse BM25 encoding
- Reciprocal Rank Fusion algorithmic ordering
- In-memory Qdrant collection upsert & hybrid retrieval
- Multi-provider LLM grounding & extractive fallback
- End-to-end FastAPI endpoint workflows (upload -> search -> query -> delete)

---

## 📄 License
MIT License. Free for commercial and non-commercial use.
