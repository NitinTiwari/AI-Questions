"""
FastAPI application for Multimodal Agentic RAG over Word documents with PostgreSQL.
Provides endpoints for question-answering, semantic retrieval, document re-indexing,
static image serving, and an embedded modern Chat UI for end users.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
    tokens_used: Optional[int] = Field(default=None, description="Estimated total tokens used")
    latency_ms: Optional[float] = Field(default=None, description="Execution latency in milliseconds")


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
    """Health check, database collection statistics, cache metrics, and LiteLLM gateway status."""
    from src.vector_store import get_collection_count
    from src.cache import get_cache_stats
    from src.gateway import get_gateway_stats
    return {
        "status": "online",
        "app": "Multimodal Agentic RAG API",
        "docs_url": "/docs",
        "collection": settings.PG_COLLECTION_NAME,
        "total_vectors": get_collection_count(),
        "cache": get_cache_stats(),
        "gateway": get_gateway_stats(),
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
        err_str = str(e)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            raise HTTPException(
                status_code=429,
                detail="Groq token rate limit reached. Please wait 10-15 seconds and try again.",
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask/stream", tags=["Streaming"])
async def ask_question_stream(request: AskRequest, req: Request):
    """
    Server-Sent Events (SSE) streaming endpoint for Real-Time Q&A.
    Emits status transitions, token-by-token generation chunks, and final metadata.
    """
    base_url = str(req.base_url).rstrip("/")

    async def sse_generator():
        import json
        import time
        from src.cache import get_cached_response, set_cached_response, normalize_query
        from src.gateway import get_gateway_llm, _metrics
        from src.hybrid_retriever import get_hybrid_retriever

        # Ensure tools/retriever are initialized
        get_agent()
        start_time = time.time()
        norm_q = normalize_query(request.question)

        # Step 1: Query Ingestion
        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 1, 'name': 'Query Ingestion', 'status': 'completed', 'detail': f'Normalized query: \"{norm_q[:60]}\"', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

        # Step 2: Check L1 / L2 Caches
        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 2, 'name': 'Cache Inspection', 'status': 'running', 'detail': 'Checking L1 In-Memory TTL & L2 pgvector semantic cache...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'stage': 'cache_check', 'message': 'Checking hybrid cache...'})}\n\n"
        
        cached = get_cached_response(request.question)
        if cached:
            tier_code = cached.get("tier")
            is_l1 = (tier_code == "in_memory")
            tier_label = "Tier 1 (L1) In-Memory TTLCache" if is_l1 else f"Tier 2 (L2) pgvector Semantic Cache (sim: {round(cached.get('similarity', 0.95)*100, 1)}%)"
            elapsed_cache_ms = round((time.time() - start_time) * 1000, 1)

            yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 2, 'name': 'Cache Inspection', 'status': 'hit', 'tier': tier_code, 'tier_label': tier_label, 'detail': f'HIT in {tier_label} ({elapsed_cache_ms}ms)', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
            yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 3, 'name': 'LLM Gateway', 'status': 'skipped', 'detail': 'Bypassed external LLM API & DB search (served directly from cache)', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
            yield f"data: {json.dumps({'type': 'cache_hit', 'tier': tier_code, 'similarity': cached.get('similarity'), 'latency_ms': elapsed_cache_ms})}\n\n"
            
            answer_text = cached["answer"]
            words = answer_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.012)

            total_elapsed = round((time.time() - start_time) * 1000, 1)
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_text, 'images': cached.get('images', []), 'cached': True, 'tier': tier_code, 'tokens': 0, 'latency_ms': total_elapsed})}\n\n"
            return

        # Cache Miss
        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 2, 'name': 'Cache Inspection', 'status': 'miss', 'detail': 'Cache Miss in L1 & L2 (escalating to retrieval & LiteLLM Gateway)', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

        # Step 3: Hybrid Search Retrieval (BM25 + pgvector + Re-ranking)
        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 3, 'name': 'Hybrid Search Retrieval', 'status': 'running', 'detail': 'Dense pgvector (HNSW) + Sparse BM25 + Reciprocal Rank Fusion + Re-Ranking...', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieving', 'message': 'Searching document with Hybrid Search (BM25 + pgvector + Re-ranking)...'})}\n\n"
        
        hybrid = get_hybrid_retriever()
        if hybrid is not None:
            retrieved_docs = hybrid.retrieve(request.question, top_k=settings.RETRIEVAL_TOP_K)
        else:
            from src.vector_store import get_vector_store
            vs = get_vector_store()
            retrieved_docs = vs.similarity_search(request.question, k=settings.RETRIEVAL_TOP_K)

        images = []
        context_passages = []
        for i, doc in enumerate(retrieved_docs, start=1):
            doc_type = doc.metadata.get("type", "text")
            sec = doc.metadata.get("section", "General")
            if doc_type == "image" and doc.metadata.get("image_path"):
                img_name = Path(doc.metadata["image_path"]).name
                images.append({
                    "filename": img_name,
                    "url": f"{base_url}/images/{img_name}",
                    "section": sec,
                })
            context_passages.append(f"--- Passage {i} [{sec}] ---\n{doc.page_content}")

        context_str = "\n\n".join(context_passages)
        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 3, 'name': 'Hybrid Search Retrieval', 'status': 'completed', 'detail': f'Retrieved and re-ranked top {len(retrieved_docs)} passages from AI-Notes.docx', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

        # Step 4: Stream Synthesis via LiteLLM Gateway
        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 4, 'name': 'LiteLLM Gateway Call', 'status': 'running', 'detail': f'Calling {settings.LITELLM_PRIMARY_MODEL} (Timeout: {settings.LITELLM_TIMEOUT}s, Max Tokens: {settings.LITELLM_MAX_TOKENS})', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'stage': 'generating', 'message': 'Synthesizing grounded answer with LiteLLM Gateway...'})}\n\n"
        
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_gateway_llm()
        
        prompt_messages = [
            SystemMessage(content=(
                "You are an expert AI assistant that answers questions using ONLY the provided document context passages.\n"
                "Rules:\n"
                "- Directly answer the user's question using the retrieved passages.\n"
                "- If the retrieved passages do not contain information to answer the question, clearly state: "
                "\"Based on the indexed document (AI-Notes.docx), information about this topic was not found.\"\n"
                "- Present comparisons cleanly (bullet points or tables).\n"
                "- Cite the relevant section(s) at the end."
            )),
            HumanMessage(content=f"Context Passages:\n{context_str}\n\nUser Question: {request.question}"),
        ]

        accumulated_text = ""
        try:
            for chunk in llm.stream(prompt_messages):
                token = chunk.content or ""
                if token:
                    accumulated_text += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0.005)
        except Exception as e:
            err_msg = str(e)
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                error_token = "\n\n*(Notice: Groq rate limit reached on free tier. Please wait 10-15 seconds and retry.)*"
                accumulated_text += error_token
                yield f"data: {json.dumps({'type': 'token', 'content': error_token})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"
                return

        # Calculate estimated tokens
        prompt_tokens = int(len(context_str.split()) * 1.33)
        completion_tokens = int(len(accumulated_text.split()) * 1.33)
        total_tokens = prompt_tokens + completion_tokens
        _metrics.record_call(is_fallback=False, tokens=total_tokens)

        yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 4, 'name': 'LiteLLM Gateway Call', 'status': 'completed', 'detail': f'Synthesized answer using ~{total_tokens} tokens (Prompt: ~{prompt_tokens}, Completion: ~{completion_tokens})', 'tokens': total_tokens, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

        # Step 5: Save to L1 & L2 caches on success
        if accumulated_text:
            set_cached_response(
                query=request.question,
                answer=accumulated_text,
                images=images,
            )
            yield f"data: {json.dumps({'type': 'dataflow_step', 'step': 5, 'name': 'Cache Persistence', 'status': 'completed', 'detail': f'Stored synthesized response in L1 Memory (TTL: {settings.CACHE_L1_TTL}s) & L2 PostgreSQL table (TTL: {settings.CACHE_L2_TTL}s)', 'timestamp': time.strftime('%H:%M:%S')})}\n\n"

        total_elapsed = round((time.time() - start_time) * 1000, 1)
        yield f"data: {json.dumps({'type': 'done', 'answer': accumulated_text, 'images': images, 'cached': False, 'tokens': total_tokens, 'latency_ms': total_elapsed})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/ask/stream", tags=["Streaming"])
async def ask_question_stream_get(question: str, req: Request):
    """Convenience GET endpoint for Server-Sent Events (SSE) streaming."""
    return await ask_question_stream(AskRequest(question=question), req)


@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
def search_documents(request: SearchRequest, req: Request):
    """
    Direct semantic retrieval endpoint: returns raw matching passages,
    Markdown tables, and diagram captions with static image URLs.
    """
    try:
        from src.vector_store import get_vector_store

        k = request.top_k or settings.RETRIEVAL_TOP_K
        from src.hybrid_retriever import get_hybrid_retriever
        hybrid = get_hybrid_retriever()
        if hybrid is not None:
            results = hybrid.retrieve(request.query, top_k=k)
        else:
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


@app.get("/api/gateway/stats", tags=["LLM Gateway"])
def gateway_stats():
    """Get LiteLLM gateway routing, model hierarchy, and usage statistics."""
    from src.gateway import get_gateway_stats
    return get_gateway_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
