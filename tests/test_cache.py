"""
Verification test for L1 (In-Memory) and L2 (PostgreSQL pgvector) Caching System.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import (
    normalize_query,
    ThreadSafeTTLCache,
    get_cached_response,
    set_cached_response,
    clear_all_caches,
    get_cache_stats,
)


def run_tests():
    print("=== Step 1: Testing Query Normalization ===")
    q1 = "What is HyDE??"
    q2 = "  what is hyde   "
    q3 = "What is HYDE."
    assert normalize_query(q1) == "what is hyde", f"Failed: {normalize_query(q1)}"
    assert normalize_query(q2) == "what is hyde", f"Failed: {normalize_query(q2)}"
    assert normalize_query(q3) == "what is hyde", f"Failed: {normalize_query(q3)}"
    print("[PASS] Normalization works identically across variations.")

    print("\n=== Step 2: Testing L1 ThreadSafeTTLCache ===")
    ttl_cache = ThreadSafeTTLCache(maxsize=3, ttl=1)
    ttl_cache.set("k1", {"ans": "v1"})
    ttl_cache.set("k2", {"ans": "v2"})
    assert ttl_cache.get("k1")["ans"] == "v1"
    assert ttl_cache.stats()["hits"] == 1
    assert ttl_cache.stats()["size"] == 2
    
    # Test TTL expiration
    time.sleep(1.2)
    assert ttl_cache.get("k1") is None
    print("[PASS] L1 ThreadSafeTTLCache expiration and LRU work correctly.")

    print("\n=== Step 3: Clearing existing test caches ===")
    clear_all_caches()

    print("\n=== Step 4: Testing L1 & L2 Cache Storage ===")
    test_question = "What is the HyDE retrieval architecture?"
    test_answer = "HyDE (Hypothetical Document Embeddings) generates a pseudo-document from an LLM and embeds it for vector retrieval."
    test_images = [{"filename": "image_01.png", "url": "/images/image_01.png", "section": "Retrieval"}]

    set_cached_response(test_question, test_answer, test_images)
    print("[PASS] Cached response stored in both L1 and L2.")

    print("\n=== Step 5: Testing L1 Exact Match Hit ===")
    # Exact normalized match
    res_l1 = get_cached_response("what is the hyde retrieval architecture?")
    assert res_l1 is not None, "Expected L1 cache hit"
    assert res_l1["tier"] == "in_memory", f"Expected in_memory tier, got {res_l1['tier']}"
    assert res_l1["similarity"] == 1.0
    assert res_l1["answer"] == test_answer
    assert len(res_l1["images"]) == 1
    print(f"[PASS] L1 Hit confirmed! Tier: {res_l1['tier']}, Similarity: {res_l1['similarity']}")

    print("\n=== Step 6: Testing L2 Semantic Similarity Hit (Paraphrased Query) ===")
    # Clear L1 to force L2 check
    from src.cache import _l1_cache
    _l1_cache.clear()

    # Paraphrased question that hasn't been in L1
    paraphrased_q = "Tell me about the HyDE retrieval architecture"
    res_l2 = get_cached_response(paraphrased_q, similarity_threshold=0.90)
    assert res_l2 is not None, "Expected L2 semantic pgvector hit"
    assert res_l2["tier"] == "pgvector", f"Expected pgvector tier, got {res_l2['tier']}"
    assert res_l2["similarity"] >= 0.85, f"Expected similarity >= 0.85, got {res_l2['similarity']}"
    assert res_l2["answer"] == test_answer
    print(f"[PASS] L2 Semantic Hit confirmed! Tier: {res_l2['tier']}, Similarity: {res_l2['similarity']}")

    # Verify that L2 hit also populated L1 for next time
    res_re_l1 = get_cached_response(paraphrased_q)
    assert res_re_l1 is not None and res_re_l1["tier"] == "in_memory"
    print("[PASS] L2 hit successfully populated L1 for future instant calls.")

    print("\n=== Step 7: Testing Cache Stats ===")
    stats = get_cache_stats()
    print("Stats output:", stats)
    assert stats["cache_enabled"] is True
    assert stats["l2_pgvector"]["cached_entries"] >= 1

    print("\n=== Step 8: Testing Cache Flush ===")
    flush_res = clear_all_caches()
    assert flush_res["status"] == "success"
    assert get_cached_response(test_question) is None
    print("[PASS] Cache successfully flushed.")

    print("\n>>> ALL CACHE TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_tests()
