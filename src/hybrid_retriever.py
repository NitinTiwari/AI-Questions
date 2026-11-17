"""
Hybrid Search Retriever combining:
1. Dense Vector Retrieval (PostgreSQL pgvector / HNSW index)
2. Sparse Lexical Search (BM25 / rank_bm25)
3. Reciprocal Rank Fusion (RRF) for ranking harmonization
4. Optional Cross-Encoder Re-Ranking Stage (with graceful fallback)
"""
import logging
import threading
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReRanker:
    """
    Cross-Encoder Re-Ranker.
    If feasible and enabled, scores candidate (query, passage) pairs
    using full cross-attention. If loading fails or is disabled,
    gracefully falls back to RRF rankings without blocking.
    """

    def __init__(self, model_name: str, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def _get_model(self):
        if not self.enabled or self._failed:
            return None
        if self._model is None:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        from sentence_transformers import CrossEncoder
                        logger.info(f"Loading CrossEncoder re-ranker: '{self.model_name}'...")
                        self._model = CrossEncoder(self.model_name, max_length=512)
                        logger.info("CrossEncoder re-ranker loaded successfully.")
                    except Exception as e:
                        logger.warning(
                            f"CrossEncoder could not be loaded ({e}). "
                            "Proceeding with pure Reciprocal Rank Fusion (RRF)."
                        )
                        self._failed = True
                        self._model = None
        return self._model

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        Re-ranks candidate documents against query using CrossEncoder if available,
        otherwise preserves original order and returns top_k.
        """
        if not documents:
            return []

        model = self._get_model()
        if model is None:
            return documents[:top_k]

        try:
            pairs = [(query, doc.page_content) for doc in documents]
            scores = model.predict(pairs)
            
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata),
                )
                doc_copy.metadata["rerank_score"] = float(score)
                scored_docs.append(doc_copy)

            # Sort descending by re-ranking score
            scored_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
            logger.info(
                f"[ReRanker] Re-ranked {len(documents)} candidates -> returning top {top_k} (top score: {scored_docs[0].metadata.get('rerank_score'):.3f})"
            )
            return scored_docs[:top_k]
        except Exception as e:
            logger.warning(f"Re-ranking failed ({e}), returning RRF candidates.")
            return documents[:top_k]


class HybridRetriever:
    """
    Unified Hybrid Search Retriever:
    Combines dense pgvector similarity with sparse BM25 lexical retrieval,
    harmonizes rankings via Reciprocal Rank Fusion (RRF), and applies an optional
    Cross-Encoder re-ranking stage.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._sparse_retriever: Optional[BM25Retriever] = None
        self._lock = threading.Lock()
        self.reranker = CrossEncoderReRanker(
            model_name=settings.RERANKER_MODEL,
            enabled=settings.RERANKING_ENABLED,
        )

    def _get_sparse_retriever(self) -> Optional[BM25Retriever]:
        """Lazily initialize BM25 index over documents."""
        if self._sparse_retriever is None:
            with self._lock:
                if self._sparse_retriever is None:
                    try:
                        from src.document_loader import load_documents
                        from src.chunker import chunk_documents

                        logger.info("Building BM25 sparse index over document chunks...")
                        docs = load_documents()
                        chunks = chunk_documents(docs)
                        self._sparse_retriever = BM25Retriever.from_documents(chunks)
                        self._sparse_retriever.k = settings.RETRIEVAL_TOP_K * 2
                        logger.info(f"BM25 sparse index ready ({len(chunks)} chunks).")
                    except Exception as e:
                        logger.warning(f"Could not build BM25 sparse retriever: {e}")
                        self._sparse_retriever = None
        return self._sparse_retriever

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Executes hybrid search (dense + sparse BM25 + RRF) with optional cross-encoder re-ranking.
        """
        k = top_k or settings.RETRIEVAL_TOP_K
        candidate_pool_size = max(k * 2, 8)

        # 1. Dense Vector Search (pgvector)
        dense_docs = []
        try:
            dense_docs = self.vector_store.similarity_search(query, k=candidate_pool_size)
        except Exception as e:
            logger.warning(f"Dense vector retrieval failed: {e}")

        # If hybrid search is disabled, return dense results directly
        if not settings.HYBRID_SEARCH_ENABLED:
            return dense_docs[:k]

        # 2. Sparse Lexical Search (BM25)
        sparse_retriever = self._get_sparse_retriever()
        sparse_docs = []
        if sparse_retriever is not None:
            try:
                sparse_docs = sparse_retriever.invoke(query)[:candidate_pool_size]
            except Exception as e:
                logger.warning(f"Sparse BM25 retrieval failed: {e}")

        # If one retriever returned nothing, fallback to the other
        if not sparse_docs:
            return self.reranker.rerank(query, dense_docs, k)
        if not dense_docs:
            return self.reranker.rerank(query, sparse_docs, k)

        # 3. Reciprocal Rank Fusion (RRF)
        # Formula: Score(d) = sum( w / (60 + rank) )
        rrf_constant = 60
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key = doc.page_content.strip()
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (0.5 / (rrf_constant + rank))

        for rank, doc in enumerate(sparse_docs, start=1):
            key = doc.page_content.strip()
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (0.5 / (rrf_constant + rank))

        # Sort candidate documents by RRF score descending
        fused_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_candidates = [doc_map[k_val] for k_val in fused_keys]

        for d in fused_candidates:
            d_key = d.page_content.strip()
            d.metadata["rrf_score"] = round(rrf_scores.get(d_key, 0.0), 5)

        logger.info(
            f"[Hybrid Search] Dense: {len(dense_docs)}, Sparse: {len(sparse_docs)} -> "
            f"Fused Candidates: {len(fused_candidates)}"
        )

        # 4. Cross-Encoder Re-Ranking Stage
        final_docs = self.reranker.rerank(query, fused_candidates, k)
        return final_docs


_hybrid_retriever: Optional[HybridRetriever] = None


def init_hybrid_retriever(vector_store) -> HybridRetriever:
    """Initialize the singleton HybridRetriever with the given vector store."""
    global _hybrid_retriever
    _hybrid_retriever = HybridRetriever(vector_store)
    return _hybrid_retriever


def get_hybrid_retriever() -> Optional[HybridRetriever]:
    """Return the active singleton HybridRetriever."""
    return _hybrid_retriever
