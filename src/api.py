"""
FastAPI application for Multimodal Agentic RAG over Word documents with PostgreSQL.
Provides endpoints for question-answering, semantic retrieval, document re-indexing,
static image serving, and an embedded modern Chat UI for end users.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Multimodal RAG API",
    description="Agentic Retrieval-Augmented Generation API over Word documents with PostgreSQL (pgvector).",
    version="1.0.0",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to static frontend files
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount extracted images folder as static files so frontend can display them directly
if settings.EXTRACTED_IMAGES_PATH.exists():
    app.mount(
        "/images",
        StaticFiles(directory=str(settings.EXTRACTED_IMAGES_PATH)),
        name="images",
    )

# Lazily initialized global agent executor
_agent_executor = None


def get_agent():
    global _agent_executor
    if _agent_executor is None:
        from src.agent import build_agent
        settings.validate()
        _agent_executor = build_agent(force_rebuild_index=False)
    return _agent_executor


# --- Pydantic Request & Response Models ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for the agent", example="What is vibe coding?")


class ImageItem(BaseModel):
    filename: str
    url: str
    section: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    images: List[ImageItem] = Field(default_factory=list, description="Diagrams/images referenced in retrieved context")
    cached: bool = Field(default=False, description="Whether this response was served from cache")
    cache_tier: Optional[str] = Field(default=None, description="'in_memory' or 'pgvector' if cached")
    similarity: Optional[float] = Field(default=None, description="Similarity score if retrieved from semantic cache")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query for similarity search", example="HyDE architecture")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of results to return")


class SearchItem(BaseModel):
    content: str
    type: str
    section: str
    source: str
    image_url: Optional[str] = None
    table_index: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchItem]


class ReindexResponse(BaseModel):
    status: str
    message: str
    total_vectors: int


# --- API Endpoints ---

@app.get("/", tags=["UI"])
def serve_ui():
    """Serves the Chat UI for end users, or JSON metadata if UI is missing."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    from src.vector_store import get_collection_count
    return {
        "status": "online",
        "app": "Multimodal Agentic RAG API",
        "docs_url": "/docs",
        "collection": settings.PG_COLLECTION_NAME,
        "total_vectors": get_collection_count(),
    }


@app.get("/api/health", tags=["Health"])
def health():
    """Health check, database collection statistics, and cache metrics."""
    from src.vector_store import get_collection_count
    from src.cache import get_cache_stats
    return {
        "status": "online",
        "app": "Multimodal Agentic RAG API",
        "docs_url": "/docs",
        "collection": settings.PG_COLLECTION_NAME,
        "total_vectors": get_collection_count(),
        "cache": get_cache_stats(),
    }


@app.post("/api/ask", response_model=AskResponse, tags=["RAG Agent"])
def ask_question(request: AskRequest, req: Request):
    """
    Query the tool-calling agent. Checks L1 In-Memory and L2 PostgreSQL pgvector
    caches first for instantaneous responses. Synthesizes via Groq on cache miss.
    """
    try:
        from src.cache import get_cached_response, set_cached_response

        # 1. Check L1 / L2 Caches first
        cached = get_cached_response(request.question)
        if cached:
            img_items = [
                ImageItem(
                    filename=img.get("filename", ""),
                    url=img.get("url", ""),
                    section=img.get("section"),
                )
                for img in cached.get("images", [])
            ]
            return AskResponse(
                question=request.question,
                answer=cached["answer"],
                images=img_items,
                cached=True,
                cache_tier=cached.get("tier"),
                similarity=cached.get("similarity"),
            )

        # 2. Cache Miss: Execute Groq Agent & PGVector retrieval
        from src.agent import ask
        from src.vector_store import get_vector_store

        agent = get_agent()
        answer = ask(agent, request.question)

        # Detect any diagrams relevant to this question for rich frontend presentation
        vs = get_vector_store()
        top_matches = vs.similarity_search(request.question, k=3)
        images = []
        base_url = str(req.base_url).rstrip("/")

        for doc in top_matches:
            if doc.metadata.get("type") == "image":
                img_path_str = doc.metadata.get("image_path")
                if img_path_str:
                    img_name = Path(img_path_str).name
                    images.append(
                        ImageItem(
                            filename=img_name,
                            url=f"{base_url}/images/{img_name}",
                            section=doc.metadata.get("section"),
                        )
                    )

        # 3. Save to L1 & L2 caches
        set_cached_response(
            query=request.question,
            answer=answer,
            images=[img.dict() for img in images],
        )

        return AskResponse(
            question=request.question,
            answer=answer,
            images=images,
            cached=False,
        )
    except Exception as e:
        logger.exception("Error processing question in /api/ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
def search_documents(request: SearchRequest, req: Request):
    """
    Direct semantic retrieval endpoint: returns raw matching passages,
    Markdown tables, and diagram captions with static image URLs.
    """
    try:
        from src.vector_store import get_vector_store

        k = request.top_k or settings.RETRIEVAL_TOP_K
        vs = get_vector_store()
        results = vs.similarity_search(request.query, k=k)
        base_url = str(req.base_url).rstrip("/")

        items = []
        for doc in results:
            doc_type = doc.metadata.get("type", "text")
            image_url = None
            if doc_type == "image" and doc.metadata.get("image_path"):
                img_name = Path(doc.metadata["image_path"]).name
                image_url = f"{base_url}/images/{img_name}"

            items.append(
                SearchItem(
                    content=doc.page_content,
                    type=doc_type,
                    section=doc.metadata.get("section", "General"),
                    source=doc.metadata.get("source", "unknown"),
                    image_url=image_url,
                    table_index=doc.metadata.get("table_index"),
                )
            )

        return SearchResponse(
            query=request.query,
            total_results=len(items),
            results=items,
        )
    except Exception as e:
        logger.exception("Error processing search in /api/search")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex", response_model=ReindexResponse, tags=["Admin"])
def reindex_document():
    """
    Re-parses the source Word document(s), converts tables to Markdown,
    extracts/captions images, updates PostgreSQL pgvector, and flushes all caches.
    """
    global _agent_executor
    try:
        from src.vector_store import get_or_build_vector_store, get_collection_count
        from src.tools import init_tools
        from src.cache import clear_all_caches

        vs = get_or_build_vector_store(force_rebuild=True)
        count = get_collection_count()
        # Re-initialize the agent retriever with updated store
        init_tools(vs)
        _agent_executor = None

        # Invalidate caches on document rebuild
        clear_all_caches()

        return ReindexResponse(
            status="success",
            message="Document parsed and re-indexed into PostgreSQL successfully. All caches cleared.",
            total_vectors=count,
        )
    except Exception as e:
        logger.exception("Error during reindex in /api/reindex")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache/stats", tags=["Cache"])
def cache_stats():
    """Get L1 In-Memory and L2 PostgreSQL pgvector cache statistics."""
    from src.cache import get_cache_stats
    return get_cache_stats()


@app.post("/api/cache/clear", tags=["Cache"])
def clear_cache():
    """Flush both L1 In-Memory and L2 PostgreSQL pgvector caches."""
    from src.cache import clear_all_caches
    return clear_all_caches()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
