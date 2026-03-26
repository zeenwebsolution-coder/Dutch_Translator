"""
rag_engine.py — LangChain-based RAG engine for Dutch tone retrieval.

Pipeline:
  1. Chunk raw tone text using LangChain RecursiveCharacterTextSplitter.
  2. Embed every chunk with LangChain OpenAIEmbeddings (text-embedding-3-small).
  3. Store in a FAISS in-memory vector store.
  4. At query time, retrieve the top-k most relevant tone chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
)

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RAGStore:
    """Wraps a LangChain FAISS vector store for tone retrieval."""
    vectorstore: Optional[object] = None
    chunk_count: int = 0

    @property
    def is_ready(self) -> bool:
        return self.vectorstore is not None and self.chunk_count > 0


# ── Public API ────────────────────────────────────────────────────────────────

def build_rag_store(tone_text: str, api_key: str) -> RAGStore:
    """
    Chunk the tone text, embed every chunk with LangChain, and return a RAGStore.

    Args:
        tone_text: Raw Dutch reference text.
        api_key:   OpenAI API key for embeddings.

    Returns:
        A RAGStore ready for retrieval.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    if not tone_text or not tone_text.strip():
        return RAGStore()

    # 1. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_text(tone_text)

    if not chunks:
        return RAGStore()

    logger.info("Split tone text into %d chunks", len(chunks))

    # 2. Create embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    # 3. Build FAISS vector store from chunks
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )

    return RAGStore(vectorstore=vectorstore, chunk_count=len(chunks))


def retrieve_tone_context(
    words: list[str],
    store: RAGStore,
    top_k: int = TOP_K_CHUNKS,
) -> str:
    """
    Given a list of English words (a translation batch), retrieve the
    most semantically relevant Dutch tone chunks from the store.

    Args:
        words:  English words/phrases in the current batch.
        store:  Populated RAGStore from build_rag_store().
        top_k:  Maximum number of tone chunks to return.

    Returns:
        A single string of concatenated relevant Dutch tone excerpts,
        ready to be injected into the GPT-4 system prompt.
        Returns "" if the store is empty.
    """
    if not store.is_ready:
        return ""

    query = " ".join(words)

    try:
        docs = store.vectorstore.similarity_search(query, k=top_k)
        excerpts = [doc.page_content for doc in docs]
        return "\n".join(excerpts)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return ""
