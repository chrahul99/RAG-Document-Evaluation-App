import re
import time
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from utils.config import AppConfig
from utils.logger import get_logger


logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Normalize PDF text while preserving paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> list[Document]:
    """Extract page-level text from a PDF file."""
    documents: list[Document] = []
    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            continue

        documents.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_path.name,
                    "file_path": str(pdf_path),
                    "page": page_number,
                },
            )
        )

    return documents


def chunk_documents(documents: Iterable[Document], config: AppConfig) -> list[Document]:
    """Split documents into overlapping chunks for semantic retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}-{index}"
        chunk.metadata["chunk_index"] = index

    return chunks


def process_pdf_files(pdf_paths: list[Path], config: AppConfig) -> dict:
    """Extract and chunk a batch of PDF files."""
    start_time = time.perf_counter()
    all_pages: list[Document] = []
    file_summaries = []

    for pdf_path in pdf_paths:
        try:
            pages = extract_pdf_text(pdf_path)
            all_pages.extend(pages)
            file_summaries.append(
                {
                    "file_name": pdf_path.name,
                    "pages_extracted": len(pages),
                    "status": "processed",
                }
            )
            logger.info("Processed %s with %s pages", pdf_path.name, len(pages))
        except Exception as exc:
            logger.exception("Failed to process %s", pdf_path)
            file_summaries.append(
                {
                    "file_name": pdf_path.name,
                    "pages_extracted": 0,
                    "status": f"failed: {exc}",
                }
            )

    chunks = chunk_documents(all_pages, config)
    elapsed = time.perf_counter() - start_time

    return {
        "chunks": chunks,
        "file_summaries": file_summaries,
        "chunk_count": len(chunks),
        "processing_time_seconds": elapsed,
    }
