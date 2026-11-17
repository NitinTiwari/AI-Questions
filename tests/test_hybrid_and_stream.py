"""
Test suite verifying:
1. L2 Vector Semantic Cache TTL Validity
2. Hybrid Search (BM25 Sparse Lexical + Dense pgvector + RRF)
3. Cross-Encoder Re-Ranking Stage
4. Real-Time Streaming Responses (SSE / StreamingResponse)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.cache import purge_expired_l2_cache, get_cache_stats
from src.hybrid_retriever import HybridRetriever, CrossEncoderReRanker
from langchain_core.documents import Document
from fastapi.testclient import TestClient
from src.api import app


def test_l2_cache_ttl_config():
    print("=== Test 1: L2 Cache TTL Configuration ===")
    assert hasattr(settings, "CACHE_L2_TTL"), "settings must have CACHE_L2_TTL"
    assert settings.CACHE_L2_TTL > 0, f"Expected CACHE_L2_TTL > 0, got {settings.CACHE_L2_TTL}"
    stats = get_cache_stats()
    assert "ttl_seconds" in stats["l2_pgvector"], "L2 cache stats must report ttl_seconds"
    assert stats["l2_pgvector"]["ttl_seconds"] == settings.CACHE_L2_TTL
    purged = purge_expired_l2_cache()
    print(f"[PASS] L2 TTL configured ({settings.CACHE_L2_TTL}s). Purged expired rows: {purged}")


def test_hybrid_search_rrf():
    print("\n=== Test 2: Hybrid Search (BM25 + Dense + RRF) ===")
    # Mock vector store
    class MockVectorStore:
        def similarity_search(self, query, k=4):
            return [
                Document(page_content="Dense Match: Neural vector embeddings map words into geometric space.", metadata={"source": "test", "type": "text"}),
                Document(page_content="Dense Match: Pinecone stores dense vectors efficiently.", metadata={"source": "test", "type": "text"}),
            ]

    hybrid = HybridRetriever(MockVectorStore())
    # Retrieve using hybrid
    docs = hybrid.retrieve("neural vector embeddings", top_k=2)
    assert len(docs) > 0, "Hybrid retriever must return documents"
    print(f"[PASS] Hybrid search successfully fused candidates! Retrieved {len(docs)} documents.")
    for i, d in enumerate(docs):
        print(f"  [{i+1}] {d.page_content[:70]}... (RRF Score: {d.metadata.get('rrf_score')})")


def test_cross_encoder_reranker():
    print("\n=== Test 3: Cross-Encoder Re-Ranking Stage ===")
    reranker = CrossEncoderReRanker(model_name=settings.RERANKER_MODEL, enabled=True)
    sample_docs = [
        Document(page_content="The weather today is cloudy with light rain.", metadata={"id": 1}),
        Document(page_content="Vector embeddings are numerical representations of text capturing semantic meaning.", metadata={"id": 2}),
    ]
    ranked = reranker.rerank("What are vector embeddings?", sample_docs, top_k=2)
    assert len(ranked) == 2, "Expected 2 re-ranked documents"
    print(f"[PASS] Re-ranker completed successfully. Top choice: {ranked[0].page_content[:60]}...")


def test_sse_streaming_endpoint():
    print("\n=== Test 4: Real-Time SSE Streaming Endpoint (/api/ask/stream) ===")
    client = TestClient(app)
    # Test GET streaming endpoint with mock query
    resp = client.get("/api/ask/stream?question=What+is+vibe+coding")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "text/event-stream" in resp.headers["content-type"], "Expected text/event-stream content type"
    
    # Read first few SSE chunks
    events = [line for line in resp.text.split("\n\n") if line.startswith("data: ")]
    assert len(events) > 0, "Expected at least one SSE data event"
    print(f"[PASS] SSE streaming endpoint verified! Received {len(events)} events.")
    print("  First event:", events[0])


if __name__ == "__main__":
    test_l2_cache_ttl_config()
    test_hybrid_search_rrf()
    test_cross_encoder_reranker()
    test_sse_streaming_endpoint()
    print("\nALL HYBRID SEARCH, RE-RANKER, CACHE TTL & STREAMING TESTS PASSED!")
