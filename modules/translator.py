"""
translator.py — GPT-4o-mini translation engine with LangChain RAG & Persistent Cache.

For every batch of English words:
  1. Check persistent SQLite cache for existing translations.
  2. Retrieve relevant tone chunks via LangChain RAG for the remaining words.
  3. Call OpenAI and parse the JSON response.
  4. Update the persistent cache with new results.
"""

from __future__ import annotations

import json
import re
import time
import logging
import streamlit as st
from typing import Sequence, Dict, List, Optional

from openai import OpenAI
from modules.config import (
    TRANSLATION_MODEL,
    MAX_RETRIES,
    TEMPERATURE,
    MAX_TOKENS,
    FORMALITY_OPTIONS,
    TOP_K_CHUNKS,
)
from modules.rag_engine import RAGStore, retrieve_tone_context
from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# ── Public API ────────────────────────────────────────────────────────────────

def translate_batch(
    words: Sequence[str],
    client: OpenAI,
    rag_store: RAGStore,
    domain: str,
    formality: str,
    cache: Optional[CacheManager] = None,
) -> dict[str, str]:
    """
    Translate a batch of English words, using cache and RAG.
    """
    results: dict[str, str] = {}
    remaining_words: list[str] = []

    # 1. Check cache first
    if cache:
        for word in words:
            cached = cache.get(domain, formality, word)
            if cached:
                results[word] = cached
            else:
                remaining_words.append(word)
    else:
        remaining_words = list(words)

    if not remaining_words:
        return results

    # 2. Translate remaining words via API
    tone_context = retrieve_tone_context(remaining_words, rag_store, TOP_K_CHUNKS)
    system_prompt = _build_system_prompt(domain, formality, tone_context)
    user_msg      = _build_user_message(remaining_words)

    api_results: dict[str, str] = {}
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            api_results = _parse_json_response(raw)
            break
        except Exception as exc:
            logger.warning(f"API error on attempt {attempt + 1}: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"❌ Batch translation failed: {exc}")

    # 3. Cache and combine results
    if api_results:
        for word, dutch in api_results.items():
            if cache:
                cache.set(domain, formality, word, dutch)
            results[word] = dutch
    
    return results


def translate_single(
    word: str,
    client: OpenAI,
    rag_store: RAGStore,
    domain: str,
    formality: str,
    cache: Optional[CacheManager] = None,
) -> str:
    """Fallback single-word translation with cache check."""
    if cache:
        cached = cache.get(domain, formality, word)
        if cached:
            return cached

    tone_context  = retrieve_tone_context([word], rag_store, top_k=3)
    system_prompt = _build_system_prompt(domain, formality, tone_context)

    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f'Translate this English business word/phrase to Dutch.\n'
                               f'Reply with ONLY the translation:\n\n"{word}"'
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=60,
        )
        dutch = response.choices[0].message.content.strip().strip('"')
        if cache:
            cache.set(domain, formality, word, dutch)
        return dutch
    except Exception as exc:
        logger.error(f"Single-word failed for '{word}': {exc}")
        return word


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_system_prompt(domain: str, formality: str, tone_context: str) -> str:
    formality_note = FORMALITY_OPTIONS.get(formality, FORMALITY_OPTIONS["Neutral"])
    rag_block = f"\nTONE REFERENCE:\n\"\"\"\n{tone_context}\n\"\"\"\n" if tone_context.strip() else ""

    return f"""You are a professional Dutch business linguist.
DOMAIN: {domain}
FORMALITY: {formality_note}
{rag_block}
RULES:
1. Return ONLY a JSON mapping English -> Dutch.
2. Preserve exact capitalisation.
3. If brand/proper noun, keep English.
4. Prioritize the tone reference vocabulary.
"""


def _build_user_message(words: list[str]) -> str:
    word_list = "\n".join(f'"{w}"' for w in words)
    return (
        "Translate these words to Dutch.\n"
        "Return ONLY a JSON object.\n\n"
        + word_list
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, str]:
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw.strip())
    return {str(k): str(v) for k, v in data.items()}
