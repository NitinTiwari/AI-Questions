"""
Smart multimodal chunker:
- Preserves Markdown tables and diagram descriptions as intact atomic units.
- Splits large text blocks with RecursiveCharacterTextSplitter while
  propagating context breadcrumbs across child chunks so semantic context
  is never lost.
- Tags every chunk with stable metadata (chunk_id, type, section, etc.).
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []

    for doc in documents:
        doc_type = doc.metadata.get("type", "text")

        # Keep tables and image descriptions as atomic units whenever possible
        if doc_type in ("table", "image"):
            # If a table is gigantic (> 3000 chars), split lines, otherwise keep intact
            if len(doc.page_content) <= settings.CHUNK_SIZE * 3:
                all_chunks.append(doc)
                continue

        section = doc.metadata.get("section", "")
        context_prefix = f"[Context: {section}]\n\n" if section else ""

        content = doc.page_content
        if context_prefix and content.startswith(context_prefix):
            content_to_split = content[len(context_prefix):]
        else:
            content_to_split = content

        sub_texts = text_splitter.split_text(content_to_split)
        if not sub_texts:
            continue

        for idx, sub_t in enumerate(sub_texts):
            chunk_content = f"{context_prefix}{sub_t}".strip()
            chunk_meta = dict(doc.metadata)
            chunk_meta["sub_chunk_index"] = idx
            all_chunks.append(Document(page_content=chunk_content, metadata=chunk_meta))

    # Stable sequential chunk_id tag
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    return all_chunks
