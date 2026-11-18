# Multimodal Agentic RAG over Word Documents with PostgreSQL (pgvector)

An enterprise-grade agentic Retrieval-Augmented Generation (RAG) system that parses complex Microsoft Word documents (`.docx`), extracts text with hierarchical breadcrumb context (`[Context: H1 > H2]`), preserves structured comparison tables (as GitHub-flavored Markdown), and extracts/captions embedded technical architecture diagrams.

Vectors are stored in **PostgreSQL (`pgvector`)** with an **HNSW index**, reasoned over by an autonomous tool-calling agent governed by an embedded **LiteLLM Gateway** (primary + fallback models, token caps, timeouts), accelerated by a **2-Tier Hybrid Cache** (L1 In-Memory + L2 pgvector Semantic), and served via a high-performance **FastAPI** backend with an **embedded single-page web UI**.

---

## Key Features

- **Multimodal Word Parsing**: Extracts text paragraphs with parent heading breadcrumbs, 24 multi-column comparison tables into Markdown, and 36 embedded architecture diagrams.
- **Multimodal Vision Captioning**: Translates visual system flowcharts and schematics into dense technical descriptions using Groq Vision models, cached locally in JSON.
- **Hybrid Search (Dense + Sparse BM25 + RRF)**: Combines dense vector retrieval (`pgvector` HNSW cosine) with sparse lexical search (`rank_bm25` Okapi) fused via Reciprocal Rank Fusion (RRF constant $k=60$). Resolves exact keyword matches and vocabulary mismatch failures.
- **Cross-Encoder Re-Ranking Stage**: Deep two-stage retrieval re-ranking top candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` for cross-attention relevance scoring, with graceful non-blocking fallback.
- **Real-Time Streaming Responses (SSE)**: Native Server-Sent Events (`StreamingResponse`) on `/api/ask/stream` streaming tokens live into the embedded UI with phase updates (`cache_check`, `retrieving`, `generating`, `token`, `done`).
- **PostgreSQL (`pgvector`) Storage**: 414 dense 384-dimensional vector embeddings (`all-MiniLM-L6-v2`) indexed with HNSW (`m=16, ef_construction=64`) using cosine distance (`<=>`).
- **Unified LiteLLM Gateway**: Primary reasoning model (`groq/qwen/qwen3.8-27b`) with automated failover to (`groq/openai/gpt-oss-20b`), hard token capping (`1500 tokens`), tool iteration limits (`max_iterations=5`), and request timeouts (`30s`).
- **2-Tier Hybrid Caching with TTL Validity**:
  - **Tier 1 (L1)**: In-Memory `TTLCache` (<1ms exact-match lookup, 2-hour TTL).
  - **Tier 2 (L2)**: PostgreSQL `pgvector` Semantic Cache (~35ms cosine similarity matching $\ge 0.78$) backed by configurable Time-To-Cache validity (`CACHE_L2_TTL = 86400s / 24h`) and automated cleanup.
  - **Self-Healing L1 Promotion**: Automatically promotes L2 hits to L1.
- **Agentic Reasoning & Tool Aliasing**: Autonomous LangChain tool-calling agent with hallucination guardrails (`search_document` and `search_function` alias).
- **FastAPI Sub-300ms Backend**: Production REST endpoints with lazy singleton initialization, CORS middleware, and static asset streaming.
- **Embedded Web UI with Left-Side Observability**: Single-page modern interface styled with Tailwind CSS, Marked.js, Lucide icons, full-screen diagram lightbox modals, and a dedicated **Left-Side Observability & Dataflow Dashboard** displaying live execution pipeline steps, separated L1 vs L2 cache hit timestamps, LiteLLM gateway tokens used, and recent query history.
- **RAGAS Evaluation Framework**: Automated quality benchmarking across Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

## Architecture & Data Flow

```
                              [data/AI-Notes.docx]
                                       │
                                       ▼
                     [src/document_loader.py (python-docx)]
     ┌─────────────────────────────────┼────────────────────────────────┐
     ▼                                 ▼                                ▼
[Text Paragraphs]              [24 Comparison Tables]         [36 Embedded Diagrams]
Dynamic Heading Stack          GitHub-Flavored Markdown       Pillow Byte Extraction
[Context: H1 > H2]             Atomic Table Preservation      Groq Vision Captioning
     │                                 │                                │
     └─────────────────────────────────┼────────────────────────────────┘
                                       │
                                       ▼
                       [src/chunker.py (Multimodal Splitter)]
                                       │
                                       ▼
                     [src/vector_store.py (langchain-postgres)]
                                       │
                                       ▼
               [PostgreSQL Database (Neon Serverless with pgvector)]
                      - 414 Chunks with HNSW Cosine Index
                      - Semantic Cache Table: rag_semantic_cache
                                       │
                                       ▼
             [REST Client / Browser / CLI: POST /api/ask or /api/ask/stream]
                                       │
                                       ▼
                       [src/cache.py (2-Tier Cache)]
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
             [Tier 1: L1 Memory]                 [Tier 2: L2 pgvector]
             MD5 Hash (<1ms hit)                 Cosine Sim >= 0.78 (~35ms)
                     │                           TTL Filter: updated_at >= NOW() - TTL
                     └─────────────────┬─────────────────┘
                                  (Cache Miss)
                                       ▼
                       [src/gateway.py (LiteLLM Gateway)]
               Primary: qwen3.8-27b  ──► Fallback: gpt-oss-20b
                                       │
                                       ▼
                         [src/agent.py (LangChain Agent)]
                     Calls `search_document` tool autonomously
                                       │
                                       ▼
                   [src/hybrid_retriever.py (Hybrid Retrieval)]
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          [Dense Vector Search]               [Sparse BM25 Search]
          pgvector HNSW Cosine                Exact Term Frequency
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                     [Reciprocal Rank Fusion (RRF k=60)]
                                       │
                                       ▼
                  [Cross-Encoder Re-Ranking Stage (Optional)]
                     ms-marco-MiniLM-L-6-v2 Cross-Attention
                                       │
                                       ▼
          [Real-Time Server-Sent Events (SSE) / JSON REST Response]
```

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **API Server** | FastAPI + Uvicorn | High-concurrency ASGI REST backend with OpenAPI/Swagger & SSE Streaming |
| **LLM Gateway** | LiteLLM | Multi-model routing, automatic failover chains, token caps, timeouts |
| **Reasoning Core** | ChatGroq / LiteLLM | Primary: `groq/qwen/qwen3.8-27b` \| Fallback: `groq/openai/gpt-oss-20b` |
| **Orchestration** | LangChain | `create_tool_calling_agent` + `AgentExecutor` |
| **Hybrid Retrieval** | `rank_bm25` + `pgvector` | Sparse BM25 + Dense vector retrieval with Reciprocal Rank Fusion (RRF) |
| **Re-Ranking** | `sentence-transformers` | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) deep relevance re-ranking |
| **Vector Database** | PostgreSQL + `pgvector` | Managed storage, HNSW cosine index, semantic cache (Neon Serverless) |
| **Embedding Model** | HuggingFace `all-MiniLM-L6-v2` | 384-dimensional dense vectors, zero API cost, local CPU inference |
| **2-Tier Caching** | Cachetools + `pgvector` | L1 In-Memory TTLCache + L2 Persistent Semantic Vector Cache with TTL validity |
| **Document Processing** | `python-docx` + `Pillow` | Multimodal OpenXML extraction (Text, Tables, Images) |
| **Vision Model** | Groq Llama 3.2 Vision | Architecture diagram captioning with persistent JSON cache |
| **Frontend UI** | HTML5 / Tailwind CSS / Marked.js | Embedded single-page interface with live SSE streaming & image lightboxes |
| **Evaluation** | RAGAS | Automated LLM-as-a-Judge benchmarking |

---

## Project Structure

```
AI-Questions/
├── .env                  # Environment variables & API credentials
├── requirements.txt      # Pinned production Python dependencies
├── Dockerfile            # Production multi-stage Docker container definition
├── data/
│   ├── AI-Notes.docx     # Source technical Word document
│   └── extracted_images/ # Extracted diagrams (image_01.png - image_36.png)
│       └── image_captions.json  # Persistent vision caption cache
├── src/
│   ├── __init__.py
│   ├── api.py            # FastAPI application & REST/SSE endpoints
│   ├── cache.py          # 2-Tier Hybrid Cache (L1 Memory + L2 pgvector with TTL)
│   ├── config.py         # Pydantic BaseSettings singleton & validation
│   ├── gateway.py        # LiteLLM Unified Gateway (Failovers, Caps, Metrics)
│   ├── hybrid_retriever.py # BM25 Sparse + Dense Retrieval + RRF + Cross-Encoder
│   ├── document_loader.py# Multimodal DOCX parser (Text, Tables, Images)
│   ├── image_captioner.py# Diagram captioning & caching
│   ├── chunker.py        # Context-preserving table & text chunking
│   ├── vector_store.py   # PostgreSQL (pgvector) storage & search
│   ├── tools.py          # Hybrid search retriever tool + aliases
│   ├── agent.py          # ChatGroq agent setup with LiteLLM gateway
│   └── main.py           # CLI entry point (Interactive, Query, Serve)
├── static/
│   └── index.html        # Embedded single-page Tailwind CSS chat interface with SSE
├── tests/
│   ├── test_cache.py     # L1 and L2 cache verification test suite
│   ├── test_gateway.py   # LiteLLM failover and token limit test suite
│   └── test_hybrid_and_stream.py # Hybrid search, Cross-Encoder, TTL & SSE tests
├── evaluation/
│   ├── __init__.py
│   ├── testset.json      # Ground-truth Q&A benchmark pairs
│   ├── ragas_eval.py     # RAGAS automated evaluation harness
│   └── results/          # Evaluation metric CSV reports
├── devDetails.txt        # Comprehensive 48-Question Developer & Interviewer Guide
└── text.txt              # Reference interview format template (Read-only)
```

---

## Quickstart

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .question

# Activate (Windows):
.question\Scripts\activate

# Activate (macOS / Linux):
source .question/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)

Create or update `.env` in the project root:

```dotenv
# Database (PostgreSQL with pgvector)
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
PG_COLLECTION_NAME=ai_notes_docs

# Groq API Credentials
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b

# Embeddings & Ingestion Configuration
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DOCS_PATH=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=4

# 2-Tier Caching Configuration
CACHE_ENABLED=true
CACHE_SEMANTIC_THRESHOLD=0.78
CACHE_L1_MAXSIZE=500
CACHE_L1_TTL=7200
CACHE_L2_TTL=86400  # L2 pgvector cache validity in seconds (24h; 0 = infinite)

# Hybrid Search & Re-Ranking Configuration
RERANKING_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Streaming Configuration
STREAMING_ENABLED=true

# LiteLLM Gateway Configuration
LITELLM_PRIMARY_MODEL=groq/qwen/qwen3.8-27b
LITELLM_FALLBACK_MODELS=groq/openai/gpt-oss-20b
LITELLM_MAX_TOKENS=1500
LITELLM_TEMPERATURE=0.0
LITELLM_TIMEOUT=30
MAX_TOOL_ITERATIONS=5
```

---

## Running the Application

### Option A: Start the FastAPI Server & Web UI

Start via the CLI:
```bash
python -m src.main --serve
```
Or directly with Uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Embedded Web Chat UI**: Navigate to `http://localhost:8000/`
- **Interactive OpenAPI Documentation (Swagger)**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

#### REST & Streaming API Endpoints:

1. **`POST /api/ask/stream` (or `GET /api/ask/stream?question=...`)**: Real-time Server-Sent Events (SSE) streaming endpoint delivering token-by-token generation with phase updates (`cache_check`, `retrieving`, `generating`, `token`, `done`).
   ```http
   POST /api/ask/stream
   Content-Type: application/json

   { "question": "Compare RAG vs Fine-Tuning" }
   ```
   *Event Stream Output:*
   ```text
   data: {"type": "status", "stage": "cache_check", "message": "Checking hybrid cache..."}
   data: {"type": "status", "stage": "retrieving", "message": "Searching documents (Hybrid BM25 + pgvector)..."}
   data: {"type": "token", "content": "RAG"}
   data: {"type": "token", "content": " and"}
   data: {"type": "token", "content": " Fine-Tuning differ..."}
   data: {"type": "done", "answer": "...", "images": [], "sources": ["AI-Notes.docx"], "cache_status": "Agent Reasoning (Hybrid + Cross-Encoder)"}
   ```

2. **`POST /api/ask`**: Full agentic question-answering with caching, diagram links, citations, and synchronous JSON payload.
   ```json
   // Request
   { "question": "Compare RAG vs Fine-Tuning in terms of Cost" }

   // Response
   {
     "question": "Compare RAG vs Fine-Tuning in terms of Cost",
     "answer": "According to the comparison table in AI-Notes.docx...",
     "images": [],
     "sources": ["AI-Notes.docx"],
     "cache_status": "Agent Reasoning (Hybrid + Cross-Encoder)"
   }
   ```

3. **`POST /api/search`**: Direct Hybrid Retrieval (BM25 sparse + pgvector dense + RRF + Cross-Encoder re-ranking).
   ```json
   // Request
   { "query": "HyDE architecture diagram", "top_k": 2 }
   ```

4. **`GET /images/{filename}`**: Static image host serving all 36 extracted diagrams (`image_01.png` - `image_36.png`).
5. **`POST /api/reindex`**: Re-parses `.docx`, updates PostgreSQL vectors, and flushes all caches.
6. **`GET /api/cache/stats`**: Real-time hit/miss rates, memory usage, L2 TTL validity, and persisted semantic cache count.
7. **`GET /api/cache/l1/entries`**: Inspect all active in-memory queries, remaining TTL countdowns, and answer previews in server RAM.
8. **`GET /api/cache/l2/entries`**: Inspect persisted semantic cache rows in PostgreSQL.
9. **`POST /api/cache/clear`**: Flushes both L1 in-memory and L2 pgvector caches.
10. **`GET /api/gateway/stats`**: LiteLLM gateway status, active models, failover counts, and token telemetry.
11. **`GET /api/health`**: Database connectivity, indexed chunk count, and system status.

---

### Option B: Command-Line Interface (CLI)

- **Interactive Q&A Loop**:
  ```bash
  python -m src.main
  ```

- **Ask a single question and exit**:
  ```bash
  python -m src.main -q "What is vibe coding?"
  ```

- **Re-index documents into PostgreSQL**:
  ```bash
  python -m src.main --rebuild-index
  ```

---

## Automated Evaluation (RAGAS)

Run automated RAG quality benchmarks against the curated testset:

```bash
python -m evaluation.ragas_eval
```

Custom testset evaluation:
```bash
python -m evaluation.ragas_eval --testset evaluation/testset.json
```

Output artifacts are generated in `evaluation/results/`:
- `ragas_eval_<timestamp>.csv`: Detailed per-question metrics.
- `ragas_eval_<timestamp>_summary.csv`: Aggregate pipeline mean scores.

Metrics Evaluated:
- **Faithfulness**: Measures absence of hallucinations (factual consistency with context).
- **Answer Relevancy**: Measures semantic relevance of the answer to the prompt.
- **Context Precision**: Measures rank quality of retrieved chunks.
- **Context Recall**: Measures completeness of retrieved context against ground truth.

---

## Running Verification Tests

Run the test suite to verify 2-Tier Caching, LiteLLM Gateway, Hybrid Search, Cross-Encoder Re-Ranking, and Real-Time SSE Streaming:

```bash
# Test 2-Tier Caching (L1 In-Memory + L2 pgvector Semantic Cache with TTL)
python tests/test_cache.py

# Test LiteLLM Gateway (Failover chain, token caps, timeouts)
python tests/test_gateway.py

# Test Hybrid Search (BM25 + pgvector + RRF), Cross-Encoder Re-Ranking, L2 Cache TTL & SSE Streaming
python tests/test_hybrid_and_stream.py
```

---

## Production Cloud Deployment (Azure)

### Deployment Architecture
- **Compute**: Azure Container Apps (ACA) or Azure App Service (Linux Containers) with auto-scaling.
- **API Gateway**: Azure API Management (APIM) for rate limiting, client subscription keys, and DDoS protection.
- **Database**: Azure Database for PostgreSQL Flexible Server with `pgvector` extension.
- **Secrets**: Azure Key Vault referenced directly into Application Settings.

### Docker Build & Run
```bash
# Build production Docker image
docker build -t multimodal-rag-app:latest .

# Run container locally
docker run -p 8000:8000 --env-file .env multimodal-rag-app:latest
```

### Azure App Service Startup Command
Under Azure Portal -> **App Service** -> **Configuration** -> **General Settings** -> **Startup Command**:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 65
```

---

## Developer Reference Manual

For a complete 50-question, 12-part architectural and functional deep dive formatted as a Senior Developer & Interviewer Reference Guide, see:
- [devDetails.txt](file:///C:/Users/hp/AI/AI-Questions/devDetails.txt)
