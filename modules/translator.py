"""
translator.py — Logic to call multi-provider LLMs (OpenAI, Claude, Gemini) 
and local NMT models (Helsinki-NLP).
Unified via LangChain's Chat Model interface or Hugging Face Pipelines.
"""

from __future__ import annotations

import json
import re
import time
import logging
import streamlit as st
from typing import Sequence, Dict, List, Optional, Any

from langchain_core.messages import SystemMessage, HumanMessage
from modules.config import (
    MAX_RETRIES,
    TOP_K_CHUNKS,
    FORMALITY_OPTIONS,
)
from modules.rag_engine import RAGStore, retrieve_tone_context
from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# ── Public API ────────────────────────────────────────────────────────────────

def translate_batch(
    words: Sequence[str],
    llm: Any,  # LangChain Chat Model OR Hugging Face Pipeline
    rag_store: RAGStore,
    domain: str,
    formality: str,
    cache: Optional[CacheManager] = None,
) -> dict[str, str]:
    """
    Translate a batch of English words via the selected provider.
    Handles both cloud LLMs and local pipelines.
    """
    results: dict[str, str] = {}
    remaining_words: list[str] = []

    # 1. Check persistent cache first
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

    # 2. Check if we are using a Local Pipeline vs Cloud LLM
    is_local = not hasattr(llm, "invoke")

    if is_local:
        # Local Model (Helsinki-NLP pipeline)
        # These models don't use 'System Prompts' or JSON.
        # We translate one by one or in a batch list.
        for word in remaining_words:
            try:
                # Prepend domain context for better NMT grounding
                context_word = f"{domain}: {word}"
                raw_translate = llm(context_word)[0]['translation_text']
                # Clean up if it repeated the domain
                clean_dutch = raw_translate.replace(f"{domain}:", "").strip()
                
                if cache:
                    cache.set(domain, formality, word, clean_dutch)
                results[word] = clean_dutch
            except Exception as e:
                logger.error(f"Local translation error: {e}")
                results[word] = word
        return results

    # 3. Cloud LLM Logic (OpenAI, Anthropic, Gemini)
    tone_context = retrieve_tone_context(remaining_words, rag_store, TOP_K_CHUNKS)
    system_prompt = _build_system_prompt(domain, formality, tone_context)
    user_msg      = _build_user_message(remaining_words)

    api_results: dict[str, str] = {}
    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg),
            ]
            response = llm.invoke(messages)
            raw = str(response.content).strip()
            api_results = _parse_json_response(raw)
            break
        except Exception as exc:
            logger.warning(f"LLM API error (attempt {attempt + 1}): {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"❌ Translation failed: {exc}")

    if api_results:
        for word, dutch in api_results.items():
            if cache:
                cache.set(domain, formality, word, dutch)
            results[word] = dutch
    
    return results


def translate_single(
    word: str,
    llm: Any,
    rag_store: RAGStore,
    domain: str,
    formality: str,
    cache: Optional[CacheManager] = None,
) -> str:
    """Fallback single-word translation via any provider."""
    if cache:
        cached = cache.get(domain, formality, word)
        if cached:
            return cached

    is_local = not hasattr(llm, "invoke")
    if is_local:
        try:
            dutch = llm(word)[0]['translation_text'].strip()
            if cache:
                cache.set(domain, formality, word, dutch)
            return dutch
        except:
            return word

    tone_context  = retrieve_tone_context([word], rag_store, top_k=3)
    system_prompt = _build_system_prompt(domain, formality, tone_context)

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'Translate ONLY the word "{word}" to Dutch.'),
        ]
        response = llm.invoke(messages)
        dutch = str(response.content).strip().strip('"').strip("'")
        if cache:
            cache.set(domain, formality, word, dutch)
        return dutch
    except:
        return word


# ── Prompt construction (for Cloud LLMs only) ─────────────────────────────────

def _build_system_prompt(domain: str, formality: str, tone_context: str) -> str:
    formality_note = FORMALITY_OPTIONS.get(formality, "Formal (u-form)")
    rag_block = f"\nTONE REFERENCE:\n\"\"\"\n{tone_context}\n\"\"\"\n" if tone_context.strip() else ""

    return f"""You are a professional Dutch business linguist.
DOMAIN: {domain}
FORMALITY: {formality_note}
{rag_block}
RULES:
1. Return ONLY a valid JSON object mapping English -> Dutch.
2. Preserve capitalization.
3. Keep brand names in English.
4. IMPORTANT: JSON format only — no markdown blocks.
"""


def _build_user_message(words: list[str]) -> str:
    word_list = "\n".join(f'"{w}"' for w in words)
    return (
        "Translate these words to Dutch. Return ONLY the JSON object:\n\n"
        + word_list
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, str]:
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.IGNORECASE)
    
    try:
        data = json.loads(raw.strip())
        return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise e
