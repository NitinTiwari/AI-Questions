"""
Hybrid Caching System for Multimodal RAG:
Tier 1 (L1): In-Memory Thread-Safe LRU + TTL Cache for instant exact-match hits (<2ms).
Tier 2 (L2): PostgreSQL pgvector Semantic Cache for paraphrased questions (cosine similarity).
"""
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import sqlalchemy
from src.config import settings

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """
    Normalizes a query string by lowercasing, stripping punctuation,
    and collapsing whitespace so that variations like 'What is HyDE?'
    and 'what is hyde' match the exact same cache key.
    """
    if not query:
        return ""
    # Strip punctuation and symbols, keep alphanumeric and whitespace
    clean = re.sub(r"[^\w\s]", "", query.lower())
    # Collapse multiple whitespaces
    return " ".join(clean.split())


class ThreadSafeTTLCache:
    """Thread-safe in-memory LRU cache with per-item Time-To-Live (TTL)."""

    def __init__(self, maxsize: int = 500, ttl: int = 7200):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[Dict[str, Any], float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            if key in self._cache:
                val, expire_at = self._cache[key]
                if now < expire_at:
                    self._cache.move_to_end(key)
                    self.hits += 1
                    return val
                else:
                    del self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Dict[str, Any]):
        now = time.time()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (value, now + self.ttl)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = round(self.hits / total, 3) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


# Singleton L1 In-Memory Cache instance
_l1_cache = ThreadSafeTTLCache(
    maxsize=settings.CACHE_L1_MAXSIZE,
    ttl=settings.CACHE_L1_TTL,
)


def _format_vector(vector: List[float]) -> str:
    """Formats a float vector into pgvector's string literal format '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def ensure_semantic_cache_table(engine: sqlalchemy.Engine):
    """Ensure the L2 pgvector semantic cache table and indexes exist in PostgreSQL."""
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(
                sqlalchemy.text("""
                    CREATE TABLE IF NOT EXISTS rag_semantic_cache (
                        id SERIAL PRIMARY KEY,
                        normalized_query TEXT NOT NULL,
                        query_embedding vector(384) NOT NULL,
                        answer TEXT NOT NULL,
                        images JSONB DEFAULT '[]'::jsonb,
                        hit_count INT DEFAULT 1,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_semantic_cache_norm_query 
                    ON rag_semantic_cache (normalized_query);
                    CREATE INDEX IF NOT EXISTS idx_rag_semantic_cache_vector 
                    ON rag_semantic_cache USING hnsw (query_embedding vector_cosine_ops);
                """)
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not verify semantic cache table: {e}")


def get_cached_response(
    query: str,
    similarity_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Attempts to retrieve a cached answer:
    1. Checks L1 In-Memory Cache (normalized query). If found -> returns in < 2ms.
    2. Checks L2 PostgreSQL pgvector Semantic Cache (cosine similarity >= threshold).
       If found -> populates L1 and returns in ~30ms.
    3. If not found -> returns None.
    """
    if not settings.CACHE_ENABLED:
        return None

    norm_q = normalize_query(query)
    if not norm_q:
        return None

    # --- Tier 1: In-Memory Cache (L1) ---
    l1_result = _l1_cache.get(norm_q)
    if l1_result is not None:
        logger.info(f"[Cache L1 HIT] Instant in-memory match for: '{norm_q[:40]}...'")
        return {
            "answer": l1_result["answer"],
            "images": l1_result["images"],
            "tier": "in_memory",
            "similarity": 1.0,
            "query": query,
        }

    # --- Tier 2: PostgreSQL Semantic Cache (L2) ---
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.CACHE_SEMANTIC_THRESHOLD
    )

    try:
        from src.vector_store import get_embeddings, get_engine

        engine = get_engine()
        embeddings = get_embeddings()
        q_emb = embeddings.embed_query(query)
        emb_str = _format_vector(q_emb)

        with engine.connect() as conn:
            # Query the closest match by cosine distance (<=>) with L2 TTL validity check
            if settings.CACHE_L2_TTL > 0:
                query_sql = sqlalchemy.text("""
                    SELECT id, normalized_query, answer, images,
                           1 - (query_embedding <=> CAST(:embedding AS vector)) AS similarity
                    FROM rag_semantic_cache
                    WHERE updated_at >= NOW() - (:ttl * INTERVAL '1 second')
                    ORDER BY query_embedding <=> CAST(:embedding AS vector) ASC
                    LIMIT 1;
                """)
                params = {"embedding": emb_str, "ttl": settings.CACHE_L2_TTL}
            else:
                query_sql = sqlalchemy.text("""
                    SELECT id, normalized_query, answer, images,
                           1 - (query_embedding <=> CAST(:embedding AS vector)) AS similarity
                    FROM rag_semantic_cache
                    ORDER BY query_embedding <=> CAST(:embedding AS vector) ASC
                    LIMIT 1;
                """)
                params = {"embedding": emb_str}

            result = conn.execute(query_sql, params).fetchone()

            if result is not None:
                sim = float(result.similarity)
                if sim >= threshold:
                    row_id = result.id
                    ans = result.answer
                    imgs = result.images
                    if isinstance(imgs, str):
                        imgs = json.loads(imgs)

                    # Increment hit counter
                    conn.execute(
                        sqlalchemy.text("""
                            UPDATE rag_semantic_cache
                            SET hit_count = hit_count + 1, updated_at = NOW()
                            WHERE id = :id;
                        """),
                        {"id": row_id},
                    )
                    conn.commit()

                    # Populate L1 cache so future identical queries hit L1
                    cached_payload = {"answer": ans, "images": imgs}
                    _l1_cache.set(norm_q, cached_payload)

                    logger.info(
                        f"[Cache L2 HIT] pgvector semantic match (similarity {sim:.3f} >= {threshold:.3f}) for: '{norm_q[:40]}...'"
                    )
                    return {
                        "answer": ans,
                        "images": imgs,
                        "tier": "pgvector",
                        "similarity": round(sim, 3),
                        "query": query,
                    }

    except Exception as e:
        logger.warning(f"Error checking L2 pgvector semantic cache: {e}")

    return None


def set_cached_response(
    query: str,
    answer: str,
    images: List[Dict[str, Any]],
):
    """
    Stores a query-answer pair in both L1 (in-memory) and L2 (PostgreSQL pgvector) caches.
    """
    if not settings.CACHE_ENABLED:
        return

    norm_q = normalize_query(query)
    if not norm_q:
        return

    payload = {"answer": answer, "images": images}

    # 1. Store in L1 immediately
    _l1_cache.set(norm_q, payload)

    # 2. Store in L2 (PostgreSQL pgvector)
    try:
        from src.vector_store import get_embeddings, get_engine

        engine = get_engine()
        embeddings = get_embeddings()
        q_emb = embeddings.embed_query(query)
        emb_str = _format_vector(q_emb)
        images_json = json.dumps(images)

        with engine.connect() as conn:
            upsert_sql = sqlalchemy.text("""
                INSERT INTO rag_semantic_cache (
                    normalized_query, query_embedding, answer, images, hit_count, updated_at
                ) VALUES (
                    :norm_q, CAST(:embedding AS vector), :answer, CAST(:images AS jsonb), 1, NOW()
                )
                ON CONFLICT (normalized_query) DO UPDATE SET
                    answer = EXCLUDED.answer,
                    images = EXCLUDED.images,
                    query_embedding = EXCLUDED.query_embedding,
                    hit_count = rag_semantic_cache.hit_count + 1,
                    updated_at = NOW();
            """)
            conn.execute(
                upsert_sql,
                {
                    "norm_q": norm_q,
                    "embedding": emb_str,
                    "answer": answer,
                    "images": images_json,
                },
            )
            conn.commit()
            logger.info(f"[Cache Saved] Cached query in L1 & L2: '{norm_q[:40]}...'")

    except Exception as e:
        logger.warning(f"Error persisting answer to L2 pgvector cache: {e}")


def clear_all_caches() -> Dict[str, Any]:
    """Flushes both L1 in-memory cache and L2 PostgreSQL table."""
    _l1_cache.clear()
    l2_deleted = 0
    try:
        from src.vector_store import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            res = conn.execute(sqlalchemy.text("TRUNCATE TABLE rag_semantic_cache;"))
            conn.commit()
            l2_deleted = 1
        logger.info("[Cache Flushed] Both L1 and L2 caches have been cleared.")
    except Exception as e:
        logger.warning(f"Error clearing L2 semantic cache: {e}")

    return {
        "status": "success",
        "message": "Both in-memory and PostgreSQL pgvector caches cleared.",
        "l1_cleared": True,
        "l2_cleared": bool(l2_deleted),
    }


def purge_expired_l2_cache() -> int:
    """
    Deletes expired rows from rag_semantic_cache whose updated_at exceeds CACHE_L2_TTL.
    Returns the number of deleted expired entries.
    """
    if settings.CACHE_L2_TTL <= 0:
        return 0

    try:
        from src.vector_store import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            res = conn.execute(
                sqlalchemy.text("""
                    DELETE FROM rag_semantic_cache
                    WHERE updated_at < NOW() - (:ttl * INTERVAL '1 second');
                """),
                {"ttl": settings.CACHE_L2_TTL},
            )
            conn.commit()
            return res.rowcount or 0
    except Exception as e:
        logger.warning(f"Error purging expired L2 cache entries: {e}")
        return 0


def get_cache_stats() -> Dict[str, Any]:
    """Returns combined performance and utilization statistics for L1 & L2."""
    l1_stats = _l1_cache.stats()
    l2_count = 0
    l2_total_hits = 0

    try:
        from src.vector_store import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                sqlalchemy.text("""
                    SELECT count(*), coalesce(sum(hit_count), 0)
                    FROM rag_semantic_cache;
                """)
            ).fetchone()
            if row:
                l2_count = row[0]
                l2_total_hits = int(row[1])
    except Exception as e:
        logger.debug(f"Could not fetch L2 cache stats: {e}")

    return {
        "cache_enabled": settings.CACHE_ENABLED,
        "semantic_threshold": settings.CACHE_SEMANTIC_THRESHOLD,
        "l1_in_memory": l1_stats,
        "l2_pgvector": {
            "cached_entries": l2_count,
            "total_hits": l2_total_hits,
            "ttl_seconds": settings.CACHE_L2_TTL,
        },
    }
