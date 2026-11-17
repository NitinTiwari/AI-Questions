"""
Builds and manages a PostgreSQL (pgvector) vector store using langchain-postgres
and HuggingFace sentence-transformer embeddings.
"""
from typing import List, Optional
import logging
import sqlalchemy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

from src.config import settings

logger = logging.getLogger(__name__)

_engine: Optional[sqlalchemy.Engine] = None


def get_engine() -> sqlalchemy.Engine:
    """Return a shared SQLAlchemy engine for PostgreSQL operations."""
    global _engine
    if _engine is None:
        _engine = sqlalchemy.create_engine(settings.DATABASE_URL)
    return _engine


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embeddings, preferring local cache to prevent DNS delays."""
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )


def get_collection_count() -> int:
    """Count how many embedding records exist for the configured collection."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            res = conn.execute(
                sqlalchemy.text("""
                    SELECT count(*) FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                    WHERE c.name = :name;
                """),
                {"name": settings.PG_COLLECTION_NAME},
            )
            return res.scalar() or 0
    except Exception as e:
        logger.debug(f"Could not retrieve collection count: {e}")
        return 0


def get_vector_store() -> PGVector:
    """Connect to the existing PostgreSQL vector store collection."""
    embeddings = get_embeddings()
    return PGVector(
        embeddings=embeddings,
        connection=settings.DATABASE_URL,
        collection_name=settings.PG_COLLECTION_NAME,
        use_jsonb=True,
    )


def build_vector_store(chunks: List[Document], force_rebuild: bool = True) -> PGVector:
    """
    Index document chunks into PostgreSQL pgvector, optionally clearing
    the existing collection first.
    """
    # Ensure vector extension is enabled
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    embeddings = get_embeddings()
    logger.info(
        f"Indexing {len(chunks)} chunks into PostgreSQL collection '{settings.PG_COLLECTION_NAME}'..."
    )

    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=settings.DATABASE_URL,
        collection_name=settings.PG_COLLECTION_NAME,
        pre_delete_collection=force_rebuild,
        use_jsonb=True,
    )
    logger.info("Successfully indexed all chunks into PostgreSQL.")
    return vector_store


def get_or_build_vector_store(force_rebuild: bool = False) -> PGVector:
    """
    Main entry point: loads existing PostgreSQL vector store if populated,
    otherwise parses the document and builds it.
    """
    settings.validate()

    if not force_rebuild:
        count = get_collection_count()
        if count > 0:
            logger.info(
                f"Using existing PostgreSQL vector store collection '{settings.PG_COLLECTION_NAME}' ({count} rows)."
            )
            return get_vector_store()

    from src.document_loader import load_documents
    from src.chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_vector_store(chunks, force_rebuild=True)
