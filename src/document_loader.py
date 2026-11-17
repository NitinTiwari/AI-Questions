"""
Unified Multimodal Document Loader for Word (.docx) documents.
Extracts text, preserves heading hierarchy, converts tables to Markdown,
and extracts/captions embedded images into structured LangChain Documents.
"""
from pathlib import Path
from typing import List, Dict
import logging

from docx import Document as DocxReader
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from langchain_core.documents import Document

from src.config import settings
from src.image_captioner import captioner

logger = logging.getLogger(__name__)


def _iter_docx_paths(source: Path) -> List[Path]:
    """Return every .docx file under `source` (or just `source` if it's a file)."""
    if source.is_file():
        if source.suffix.lower() != ".docx":
            raise ValueError(f"Not a .docx file: {source}")
        return [source]

    if not source.exists():
        raise FileNotFoundError(f"Documents path does not exist: {source}")

    docx_files = sorted(
        p for p in source.rglob("*.docx") if not p.name.startswith("~$")
    )
    if not docx_files:
        raise FileNotFoundError(f"No .docx files found under: {source}")
    return docx_files


def table_to_markdown(table) -> str:
    """Convert a python-docx Table into clean GitHub-flavored Markdown."""
    raw_rows = []
    for row in table.rows:
        row_vals = []
        for cell in row.cells:
            # Escape pipes and convert internal newlines to HTML linebreaks
            txt = cell.text.strip().replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")
            row_vals.append(txt)
        raw_rows.append(row_vals)

    if not raw_rows or not raw_rows[0]:
        return ""

    headers = raw_rows[0]
    num_cols = len(headers)
    if not any(headers):
        headers = [f"Col {i+1}" for i in range(num_cols)]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]

    for row in raw_rows[1:]:
        if len(row) < num_cols:
            row.extend([""] * (num_cols - len(row)))
        else:
            row = row[:num_cols]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _parse_single_docx(docx_path: Path) -> List[Document]:
    """
    Parse a single .docx file into structured LangChain Documents representing
    text sections, Markdown tables, and captioned images.
    """
    doc = DocxReader(str(docx_path))
    settings.EXTRACTED_IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    # Fast O(1) lookup maps
    p_map = {p._element: p for p in doc.paragraphs}
    tbl_map = {t._element: t for t in doc.tables}

    documents: List[Document] = []
    active_headings: Dict[int, str] = {}  # level -> title
    current_text_buffer: List[str] = []

    def get_current_breadcrumb() -> str:
        if not active_headings:
            return "General"
        sorted_levels = sorted(active_headings.keys())
        return " > ".join(active_headings[lvl] for lvl in sorted_levels)

    def flush_text_buffer():
        nonlocal current_text_buffer
        if not current_text_buffer:
            return
        combined_text = "\n\n".join(current_text_buffer).strip()
        if combined_text:
            breadcrumb = get_current_breadcrumb()
            documents.append(
                Document(
                    page_content=f"[Context: {breadcrumb}]\n\n{combined_text}",
                    metadata={
                        "source": docx_path.name,
                        "type": "text",
                        "section": breadcrumb,
                    },
                )
            )
        current_text_buffer = []

    img_counter = 0
    tbl_counter = 0

    all_paragraphs = doc.paragraphs
    p_to_idx = {p: i for i, p in enumerate(all_paragraphs)}

    for child in doc.element.body:
        if isinstance(child, CT_P):
            p = p_map.get(child)
            if p is None:
                continue

            style_name = (p.style.name or "").lower()
            is_heading = style_name.startswith("heading")

            if is_heading:
                flush_text_buffer()
                try:
                    level = int(style_name.replace("heading", "").strip())
                except ValueError:
                    level = 1

                heading_text = p.text.strip()
                if heading_text:
                    active_headings = {
                        lvl: title for lvl, title in active_headings.items() if lvl < level
                    }
                    active_headings[level] = heading_text

            # Check for embedded images in this paragraph
            blips = child.xpath(".//a:blip/@r:embed")
            if blips:
                flush_text_buffer()
                p_idx = p_to_idx.get(p, 0)
                self_text = p.text.strip()

                # Find nearest non-empty previous paragraph
                prev_text = self_text
                if not prev_text:
                    for back_i in range(p_idx - 1, max(-1, p_idx - 8), -1):
                        if all_paragraphs[back_i].text.strip():
                            prev_text = all_paragraphs[back_i].text.strip()
                            break

                # Find nearest non-empty subsequent paragraph
                next_text = ""
                for fwd_i in range(p_idx + 1, min(len(all_paragraphs), p_idx + 8)):
                    if all_paragraphs[fwd_i].text.strip():
                        next_text = all_paragraphs[fwd_i].text.strip()
                        break

                for r_id in blips:
                    if r_id in doc.part.rels:
                        rel = doc.part.rels[r_id]
                        if "image" in rel.target_ref:
                            img_counter += 1
                            ext = Path(rel.target_ref).suffix or ".png"
                            img_filename = f"image_{img_counter:02d}{ext}"
                            img_path = settings.EXTRACTED_IMAGES_PATH / img_filename

                            # Extract image file to disk if not exists
                            if not img_path.exists():
                                with open(img_path, "wb") as f:
                                    f.write(rel.target_part.blob)

                            breadcrumb = get_current_breadcrumb()
                            caption = captioner.generate_caption(
                                image_path=img_path,
                                section=breadcrumb,
                                prev_text=prev_text,
                                next_text=next_text,
                                source_doc=docx_path.name,
                            )

                            documents.append(
                                Document(
                                    page_content=f"[Diagram / Figure | Context: {breadcrumb}]\n{caption}",
                                    metadata={
                                        "source": docx_path.name,
                                        "type": "image",
                                        "section": breadcrumb,
                                        "image_path": str(img_path),
                                        "element_id": f"img_{img_counter:02d}",
                                    },
                                )
                            )
                continue

            # Skip if this was a heading (already recorded in active_headings)
            if is_heading:
                continue

            # Regular text paragraph
            text = p.text.strip()
            if text:
                current_text_buffer.append(text)

        elif isinstance(child, CT_Tbl):
            table = tbl_map.get(child)
            if table is None:
                continue

            flush_text_buffer()
            tbl_counter += 1
            md_table = table_to_markdown(table)

            if md_table:
                breadcrumb = get_current_breadcrumb()
                documents.append(
                    Document(
                        page_content=f"[Table | Context: {breadcrumb}]\n\n{md_table}",
                        metadata={
                            "source": docx_path.name,
                            "type": "table",
                            "section": breadcrumb,
                            "table_index": tbl_counter,
                        },
                    )
                )

    flush_text_buffer()
    logger.info(
        f"Parsed {docx_path.name}: {len(documents)} document blocks "
        f"({tbl_counter} tables, {img_counter} images)."
    )
    return documents


def load_documents(source_path: str | Path | None = None) -> List[Document]:
    """
    Load all .docx files found at `source_path` (defaults to settings.DOCS_PATH)
    into multimodal LangChain Document objects.
    """
    source = Path(source_path) if source_path else settings.DOCS_PATH
    docx_paths = _iter_docx_paths(source)

    all_docs: List[Document] = []
    for path in docx_paths:
        docs = _parse_single_docx(path)
        all_docs.extend(docs)

    return all_docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document blocks.")
    tables = [d for d in docs if d.metadata.get("type") == "table"]
    images = [d for d in docs if d.metadata.get("type") == "image"]
    texts = [d for d in docs if d.metadata.get("type") == "text"]
    print(f"  - Texts: {len(texts)}")
    print(f"  - Tables: {len(tables)}")
    print(f"  - Images: {len(images)}")
