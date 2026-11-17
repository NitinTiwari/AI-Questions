# Multimodal Agentic RAG over Word Documents with PostgreSQL (pgvector)

An agentic Retrieval-Augmented Generation (RAG) system that parses complex Word documents (`.docx`), extracts text, structured tables (Markdown), and embedded architecture diagrams, stores vector embeddings in **PostgreSQL (`pgvector`)**, and answers questions using a tool-calling reasoning agent powered by **ChatGroq**.

Now includes a production-ready **FastAPI** backend for end-user web applications, mobile apps, and frontends.

---

## Architecture & Flow

```
data/AI-Notes.docx
     │
     ▼
src/document_loader.py (python-docx)
     ├── Text Paragraphs     ──► Hierarchical Breadcrumbs ([Context: H1 > H2])
     ├── 24 Tables           ──► GitHub-flavored Markdown Tables
     └── 36 Embedded Images  ──► Extracted & Captioned (src/image_captioner.py)
     │
     ▼
src/chunker.py (Context-preserving multimodal splitter)
     │
     ▼
src/vector_store.py (langchain-postgres + sentence-transformers)
     │
     ▼
PostgreSQL Database (Neon / pgvector)
     │
     ▼
src/tools.py (search_document tool)
     │
     ▼
src/agent.py (ChatGroq tool-calling agent)
     │
     ├──► src/main.py (Interactive CLI / single query)
     └──► src/api.py  (FastAPI REST Endpoints + Static Image Serving)
```

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **API Server** | FastAPI + Uvicorn |
| **Agent / Orchestration** | LangChain (Tool-calling Agent) |
| **Reasoning LLM** | Groq (`langchain-groq`, `qwen/qwen3.8-27b`) |
| **Vector Database** | PostgreSQL with `pgvector` (`langchain-postgres`) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Caching System** | 2-Tier Hybrid: In-Memory TTLCache (L1) + PostgreSQL pgvector Semantic Cache (L2) |
| **Document Processing** | `python-docx` + `Pillow` (Custom multimodal extraction) |
| **Vision Captioning** | Groq Vision / Contextual synthesis with local JSON cache |
| **Evaluation** | RAGAS |

---

## Project Structure

```
AI-Questions/
├── .env                  # Environment variables & API keys
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container deployment definition
├── .dockerignore
├── .gitignore
├── data/
│   ├── AI-Notes.docx     # Source Word document
│   └── extracted_images/ # Extracted diagrams & captions cache
├── src/
│   ├── __init__.py
│   ├── api.py            # FastAPI application & REST endpoints
│   ├── cache.py          # 2-Tier Caching (L1 In-Memory + L2 pgvector Semantic)
│   ├── config.py         # Centralized configuration & validation
│   ├── document_loader.py# Multimodal DOCX parser (Text, Tables, Images)
│   ├── image_captioner.py# Diagram captioning & caching
│   ├── chunker.py        # Context-aware table & text chunking
│   ├── vector_store.py   # PostgreSQL (pgvector) storage & search
│   ├── tools.py          # search_document retriever tool
│   ├── agent.py          # ChatGroq agent setup
│   └── main.py           # CLI entry point
└── evaluation/
    ├── __init__.py
    ├── testset.json      # Evaluation Q&A pairs
    ├── ragas_eval.py     # RAGAS evaluation runner
    └── results/          # Evaluation metrics output
```

---

## Quickstart

### 1. Setup Virtual Environment

```bash
python -m venv .question

# Windows:
.question\Scripts\activate

# macOS / Linux:
source .question/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

```dotenv
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
PG_COLLECTION_NAME=ai_notes_docs

GROQ_API_KEY=gsk_your_groq_key
GROQ_MODEL=qwen/qwen3.8-27b

HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DOCS_PATH=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=4

# Caching Configuration (Optional)
CACHE_ENABLED=true
CACHE_SEMANTIC_THRESHOLD=0.92
CACHE_L1_MAXSIZE=500
CACHE_L1_TTL=7200
```

---

## Running the Application

### Option A: Start the FastAPI Server (For End Users)

Start via the CLI:
```bash
python -m src.main --serve
```
Or directly with Uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger):** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

#### FastAPI Endpoints:

1. **`POST /api/ask`**: Full agentic question-answering.
   ```json
   // Request
   { "question": "What is vibe coding?" }

   // Response
   {
     "question": "What is vibe coding?",
     "answer": "Vibe coding is a software development approach...",
     "images": []
   }
   ```

2. **`POST /api/search`**: Raw semantic retrieval with image URLs.
   ```json
   // Request
   { "query": "HyDE architecture diagram", "top_k": 2 }

   // Response
   {
     "query": "HyDE architecture diagram",
     "total_results": 2,
     "results": [
       {
         "type": "image",
         "section": "Advanced RAG > HyDE",
         "image_url": "http://localhost:8000/images/image_25.png",
         "content": "..."
       }
     ]
   }
   ```

3. **`GET /images/{filename}`**: Static image host serving all 36 extracted diagrams directly to web/mobile frontends.
4. **`POST /api/reindex`**: Re-parses the document, rebuilds PostgreSQL vectors, and flushes all caches.
5. **`GET /api/cache/stats`**: Detailed L1 (in-memory) and L2 (pgvector) cache hit/miss statistics.
6. **`POST /api/cache/clear`**: Explicitly flushes both L1 and L2 caches.
7. **`GET /api/health`**: Health check, database collection statistics, and cache metrics.
8. **`GET /`**: Serves the embedded single-page Chat UI.

---

### Option B: CLI Q&A

- **Interactive Q&A Loop:**
  ```bash
  python -m src.main
  ```

- **Ask a single question and exit:**
  ```bash
  python -m src.main -q "Compare RAG vs Fine-Tuning in terms of Cost"
  ```

- **Re-index documents into PostgreSQL:**
  ```bash
  python -m src.main --rebuild-index
  ```
