"""
translator.py — GPT-4o-mini translation engine with LangChain RAG tone injection.

For every batch of English words:
  1. Retrieve the most relevant tone chunks from the LangChain RAG store.
  2. Build a precise system prompt containing domain, formality, and
     the retrieved Dutch tone examples.
  3. Call GPT-4o-mini and parse the JSON response.
  4. Fall back to single-word mode if a batch call fails.
"""

from __future__ import annotations

import json
import re
import time
import logging
import streamlit as st
from typing import Sequence

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

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def translate_batch(
    words: Sequence[str],
    client: OpenAI,
    rag_store: RAGStore,
    domain: str,
    formality: str,
) -> dict[str, str]:
    """
    Translate a batch of English words to Dutch using GPT-4o-mini + RAG context.

    Args:
        words:     English words / phrases to translate.
        client:    Authenticated OpenAI client.
        rag_store: Populated RAGStore (may be empty if no tone file given).
        domain:    Business domain (e.g. "Finance & Accounting").
        formality: Key from FORMALITY_OPTIONS.

    Returns:
        Dict mapping each English word to its Dutch translation.
        On failure, returns an empty dict (caller handles fallback).
    """
    tone_context = retrieve_tone_context(list(words), rag_store, TOP_K_CHUNKS)
    system_prompt = _build_system_prompt(domain, formality, tone_context)
    user_msg      = _build_user_message(list(words))

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
            return _parse_json_response(raw)

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error on attempt %d: %s", attempt + 1, exc)
            st.warning(f"⚠️ JSON parse error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")
        except Exception as exc:
            logger.warning("API error on attempt %d: %s", attempt + 1, exc)
            st.warning(f"⚠️ API error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)   # exponential back-off

    st.error("❌ Batch translation failed after all retries. Falling back to single-word mode.")
    return {}   # signal to caller: use single-word fallback


def translate_single(
    word: str,
    client: OpenAI,
    rag_store: RAGStore,
    domain: str,
    formality: str,
) -> str:
    """
    Translate one word.  Used as the fallback when a batch call fails.

    Returns the original English word if translation is not possible.
    """
    tone_context  = retrieve_tone_context([word], rag_store, top_k=3)
    system_prompt = _build_system_prompt(domain, formality, tone_context)

    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f'Translate this English business word/phrase to Dutch.\n'
                        f'Reply with ONLY the Dutch translation — no explanation, '
                        f'no punctuation, no extra text.\n\n"{word}"'
                    ),
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as exc:
        logger.error("Single-word translation failed for '%s': %s", word, exc)
        st.error(f"❌ Failed to translate '{word}': {exc}")
        return word   # keep original on total failure


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_system_prompt(domain: str, formality: str, tone_context: str) -> str:
    formality_note = FORMALITY_OPTIONS.get(formality, FORMALITY_OPTIONS["Neutral"])

    rag_block = ""
    if tone_context.strip():
        rag_block = f"""
TONE REFERENCE — most relevant Dutch excerpts retrieved for this batch:
\"\"\"
{tone_context}
\"\"\"
Study these excerpts carefully.  Match their exact vocabulary style,
register, and terminology.  These are the gold standard for this translation.
"""

    return f"""You are a highly experienced Dutch business linguist and certified terminologist.

DOMAIN: {domain}
FORMALITY: {formality_note}
{rag_block}
TRANSLATION RULES — follow these without any exception:

1. Return ONLY a valid JSON object mapping each English input to its Dutch translation.
   Format exactly: {{"word1": "vertaling1", "word2": "vertaling2"}}
   No markdown fences, no explanations, no extra keys, no trailing commas.

2. Preserve capitalisation exactly:
   - ALL CAPS input  → ALL CAPS output
   - Title Case input → Title Case output
   - lowercase input  → lowercase output

3. For proper nouns, brand names, or abbreviations with no Dutch equivalent,
   keep the original English form unchanged.

4. Choose the Dutch term that a native Dutch-speaking professional in the
   domain of "{domain}" would immediately recognise as the correct equivalent.

5. When the tone reference is provided, prioritise vocabulary and phrasing
   consistent with that reference over generic dictionary translations.

6. Never invent translations.  If a term is genuinely untranslatable,
   return the original English word.
"""


def _build_user_message(words: list[str]) -> str:
    word_list = "\n".join(f'"{w}"' for w in words)
    return (
        "Translate each of the following English words/phrases to Dutch.\n"
        "Return ONLY a JSON object — no markdown, no extra text.\n\n"
        + word_list
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, str]:
    """Strip markdown fences and parse the JSON from GPT's response."""
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    data = json.loads(raw)
    return {str(k): str(v) for k, v in data.items()}
