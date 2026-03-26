"""
rag_engine.py — LangChain-based tone retrieval for translation.
Now supports multiple embedding providers (OpenAI, Google).
"""

from __future__ import annotations

import logging
from typing import List, Optional
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    PROVIDERS,
)

logger = logging.getLogger(__name__)

@dataclass
class RAGStore:
    vector_store: Optional[FAISS] = None
    chunk_count: int = 0

def build_rag_store(tone_text: str, provider: str, api_key: str) -> RAGStore:
    """
    Build a LangChain FAISS vector store from the tone reference text.
    Uses embeddings matching the selected provider (or OpenAI as fallback).
    """
    if not tone_text.strip():
        return RAGStore()

    # 1. Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    raw_chunks = splitter.split_text(tone_text)
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

    if not chunks:
        return RAGStore()

    # 2. Select Embeddings Model
    try:
        conf = PROVIDERS.get(provider, PROVIDERS["OpenAI"])
        embed_model = conf.get("embed_model")
        
        if provider == "Google" and embed_model:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=embed_model,
                google_api_key=api_key
            )
        else:
            # Default to OpenAI if chosen or if provider lacks native embeddings
            embeddings = OpenAIEmbeddings(
                model=PROVIDERS["OpenAI"]["embed_model"],
                api_key=api_key if provider == "OpenAI" else "" # Needs a key if used
            )
            # Special case: If user is on Anthropic, they need an OpenAI key for embeddings
            # We'll handle key availability in the UI/App layer

        # 3. Build FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)
        return RAGStore(vector_store=vector_store, chunk_count=len(chunks))

    except Exception as e:
        logger.error(f"Failed to build RAG store: {e}")
        raise e

def retrieve_tone_context(words: List[str], store: RAGStore, top_k: int = 5) -> str:
    """Retrieve the most relevant tone chunks for a given batch of words."""
    if not store.vector_store:
        return ""

    combined_query = " ".join(words)
    try:
        docs = store.vector_store.similarity_search(combined_query, k=top_k)
        return "\n---\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return ""
