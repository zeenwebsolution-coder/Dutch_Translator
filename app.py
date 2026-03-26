"""
app.py — Ultra-Simplified Dutch Business Translator.
Hardcoded for SENDERUM tone, General Business, and Formal formality.
"""

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from modules.config import compute_batch_size
from modules.tone_loader import load_tone_from_path, ToneLoadError
from modules.rag_engine import build_rag_store, RAGStore
from modules.excel_handler import (
    read_word_entries,
    write_translations,
    unique_words,
)
from modules.translator import translate_batch, translate_single
from modules.cache_manager import CacheManager
from modules.zip_handler import (
    extract_excel_files,
    pack_single,
    pack_zip,
    ZipHandlerError,
)

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()
env_api_key = os.getenv("OPENAI_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇳🇱 Translator",
    page_icon="🇳🇱",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#f8f9fb; }
.card {
    background:white; border-radius:14px;
    padding:1.8rem 2.2rem; box-shadow:0 2px 14px rgba(0,0,0,0.06);
    margin-bottom:1.4rem;
}
h1  { font-size:1.65rem !important; margin:0 !important; }
.sub { color:#6b7280; font-size:.92rem; margin:.25rem 0 .5rem; }
.step { font-size:.75rem; font-weight:700; letter-spacing:.09em;
        color:#2563eb; text-transform:uppercase; margin-bottom:.35rem; }
.warn { background:#fff7ed; border-left:4px solid #f59e0b; border-radius:6px;
        padding:.7rem 1rem; font-size:.87rem; color:#92400e; margin:.5rem 0; }
.ok   { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px;
        padding:.7rem 1rem; font-size:.87rem; color:#166534; margin:.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:.15rem">
    <span style="font-size:2rem">🇳🇱</span>
    <h1>Dutch Business Translator</h1>
  </div>
  <p class="sub">Professional English → Dutch translation using <strong>SENDERUM</strong> tone.</p>
</div>
""", unsafe_allow_html=True)

# ── Step 0: User Setup ────────────────────────────────────────────────────────
cache: CacheManager = None
tone_text: str = ""

with st.expander("👤 User Identity", expanded=True):
    col_k, col_u = st.columns([2, 1])
    with col_k:
        api_key_input = st.text_input("OpenAI API Key", value=env_api_key, type="password")
    with col_u:
        user_name = st.text_input("User Name", value="default")
        
    if api_key_input and user_name:
        cache = CacheManager(api_key_input, user_name)
        st.caption(f"💾 **Memory**: {cache.get_stats()} translations cached for **{user_name}**.")

# ── Step 1: Excel Upload ──────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 1 — Upload files</p>', unsafe_allow_html=True)
main_upload = st.file_uploader("📊 Excel file or ZIP", type=["xlsx", "zip"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Auto-Load Tone ────────────────────────────────────────────────────────────
TONE_FILE_PATH = "SENDERUM-tone of voice (1).docx"
try:
    tone_text = load_tone_from_path(TONE_FILE_PATH)
except:
    st.error("⚠️ Local tone file not found. Ensure SENDERUM file is in root.")

# ── Translation ───────────────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 2 — Professional Translation</p>', unsafe_allow_html=True)

DOMAIN = "General Business"
FORMALITY = "Formal (u-form)"
ready = bool(api_key_input and user_name and main_upload and tone_text)

if not ready:
    st.markdown('<div class="warn">🔑 Enter Identity and Upload Excel to continue.</div>', unsafe_allow_html=True)

go = st.button("🚀 Start Translation", disabled=not ready)
st.markdown('</div>', unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if go and ready:
    client = OpenAI(api_key=api_key_input)
    
    rag_store: RAGStore = RAGStore()
    with st.spinner("🔍 Building Tone Index…"):
        try:
            rag_store = build_rag_store(tone_text, api_key_input)
        except Exception as e:
            st.error(f"Error: {e}")

    try:
        excel_sources = extract_excel_files(main_upload)
        overall_bar = st.progress(0)
        translated_outputs = []
        all_summary_rows = []

        for file_idx, source in enumerate(excel_sources):
            st.info(f"Processing `{source.name}`")
            sheet_entries = read_word_entries(source.data)
            all_unique = unique_words(sheet_entries)
            batch_size = compute_batch_size(len(all_unique))

            translation_cache = {}
            processed = 0
            word_bar = st.progress(0)

            for i in range(0, len(all_unique), batch_size):
                batch = all_unique[i : i + batch_size]
                result = translate_batch(batch, client, rag_store, DOMAIN, FORMALITY, cache=cache)

                for word in batch:
                    translation_cache[word] = result.get(word, translate_single(word, client, rag_store, DOMAIN, FORMALITY, cache=cache))
                    processed += 1
                    word_bar.progress(min(processed / len(all_unique), 1.0))

            translated_bytes = write_translations(source.data, sheet_entries, translation_cache)
            translated_outputs.append((source, translated_bytes))
            overall_bar.progress((file_idx + 1) / len(excel_sources))

        st.success("✅ **Done!** Memory updated.")

        # Download
        if len(translated_outputs) == 1:
            source, t_bytes = translated_outputs[0]
            dl_bytes, dl_name = pack_single(t_bytes, source.name)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            dl_bytes, dl_name = pack_zip(translated_outputs)
            mime = "application/zip"

        st.download_button(label=f"⬇️ Download Professional Translation", data=dl_bytes, file_name=dl_name, mime=mime)

    except Exception as exc:
        st.error(str(exc))
