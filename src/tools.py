"""
Defines the tool(s) the agent can call. Currently backed by the PostgreSQL
(pgvector) retriever, structured to format multimodal context (text, Markdown
tables, and diagram/image descriptions).
"""
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from src.config import settings
from src.hybrid_retriever import init_hybrid_retriever, get_hybrid_retriever

_retriever = None  # lazily initialized, set via init_tools()


def init_tools(vector_store: VectorStore):
    """Must be called once at startup with the built/loaded vector store."""
    global _retriever
    _retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVAL_TOP_K},
    )
    init_hybrid_retriever(vector_store)


def _search_document_impl(query: str) -> str:
    hybrid = get_hybrid_retriever()
    if hybrid is not None:
        results = hybrid.retrieve(query, top_k=settings.RETRIEVAL_TOP_K)
    elif _retriever is not None:
        results = _retriever.invoke(query)
    else:
        return "Error: retriever not initialized. Call init_tools() first."

    if not results:
        return "No relevant passages found in the document."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "text").capitalize()
        section = doc.metadata.get("section", "General")

        header_parts = [f"Passage {i}", f"Type: {doc_type}", f"Section: {section}", f"Source: {source}"]
        if doc.metadata.get("image_path"):
            header_parts.append(f"Image File: {doc.metadata['image_path']}")
        if doc.metadata.get("table_index"):
            header_parts.append(f"Table #{doc.metadata['table_index']}")

        header_str = " | ".join(header_parts)
        formatted.append(f"[{header_str}]\n{doc.page_content}")

    return "\n\n".join(formatted)


@tool("search_document", response_format="content")
def search_document(query: str) -> str:
    """
    Search the indexed Word document(s) in PostgreSQL for passages, comparison tables, or
    diagrams relevant to `query`.
    Use this tool whenever you need facts, definitions, model comparisons, or
    architectural details before answering the user.
    Returns the top matching passages with their element type, section, and source.
    """
    return _search_document_impl(query)


@tool("search_function", response_format="content")
def search_function(query: str) -> str:
    """
    Alias for search_document. Search the indexed Word document(s) for passages, tables, or diagrams.
    """
    return _search_document_impl(query)


def get_tools():
    return [search_document, search_function]
