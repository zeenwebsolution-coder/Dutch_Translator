"""
app.py — Multi-Model Dutch Business Translator.

Supports OpenAI, Anthropic (Claude), and Google (Gemini).
Fixed for SENDERUM tone and General Business.
Dynamic model and API key selection.
"""

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.config import compute_batch_size, PROVIDERS, DEFAULT_PROVIDER
from modules.tone_loader import load_tone_from_path, ToneLoadError
from modules.rag_engine import build_rag_store, RAGStore
from modules.excel_handler import (
    read_word_entries,
    write_translations,
    unique_words,
)
from modules.model_factory import get_chat_model
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇳🇱 Multi-Model Translator",
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
    <h1>Multi-Model Dutch Translator</h1>
  </div>
  <p class="sub">Professional English → Dutch via <strong>OpenAI, Claude, or Gemini</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ── Step 0: Model & Identity ──────────────────────────────────────────────────
cache: CacheManager = None
tone_text: str = ""

with st.expander("🛠️ Provider & Model Setup", expanded=True):
    col_p, col_n = st.columns([1, 1])
    with col_p:
        provider = st.selectbox("Preferred Provider", list(PROVIDERS.keys()), index=0)
    with col_n:
        user_name = st.text_input("User Name", value="default")
        
    # Dynamic API Key based on provider
    key_label = f"{provider} API Key"
    env_key_name = f"{provider.upper()}_API_KEY"
    env_key_val = os.getenv(env_key_name, os.getenv("OPENAI_API_KEY", "")) if provider == "OpenAI" else ""
    
    api_key = st.text_input(key_label, value=env_key_val, type="password")
    
    # 🚨 Special Warning for Claude
    if provider == "Anthropic":
        st.caption("ℹ️ **Note**: Claude requires an **OpenAI** or **Gemini** key for RAG indexing. "
                   "If you only have a Claude key, tone reference will be skipped.")

    if api_key and user_name:
        cache = CacheManager(api_key, user_name)
        st.caption(f"💾 **Memory**: {cache.get_stats()} translations cached for **{user_name}** ({provider}).")

# ── Step 1: Excel Upload ──────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 1 — Upload files</p>', unsafe_allow_html=True)
main_upload = st.file_uploader("📊 Excel file or ZIP", type=["xlsx", "zip"])
st.markdown('</div>', unsafe_allow_html=True)

# ── Auto-Load Tone ────────────────────────────────────────────────────────────
TONE_FILE_PATH = "SENDERUM-tone of voice (1).docx"
try:
    tone_text = load_tone_from_path(TONE_FILE_PATH)
except:
    st.error("⚠️ Local tone file not found.")

# ── Defaults ──────────────────────────────────────────────────────────────────
DOMAIN = "General Business"
FORMALITY = "Formal (u-form)"
ready = bool(api_key and user_name and main_upload and tone_text)

if not ready:
    st.markdown('<div class="warn">🔑 Enter Identity and Upload Excel to continue.</div>', unsafe_allow_html=True)

go = st.button(f"🚀 Translate via {provider}", disabled=not ready)
st.markdown('</div>', unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if go and ready:
    # 1. Initialize LLM via Factory
    try:
        llm = get_chat_model(provider, api_key)
        
        # 2. Build RAG Store (Provider-Aware)
        rag_store: RAGStore = RAGStore()
        with st.spinner(f"🔍 Building {provider} Tone Index…"):
            try:
                # If Anthropic, we need a fallback key for embeddings?
                # For now, let's keep it simple: try OpenAI embeddings if key looks like OpenAI,
                # else try Google, otherwise skip.
                rag_store = build_rag_store(tone_text, provider, api_key)
            except Exception as e:
                st.warning(f"RAG Indexing skipped: {e}")

        # 3. File Loop
        excel_sources = extract_excel_files(main_upload)
        overall_bar = st.progress(0)
        translated_outputs = []
        all_summary_rows = []

        for file_idx, source in enumerate(excel_sources):
            st.info(f"Processing `{source.name}` via {provider}")
            sheet_entries = read_word_entries(source.data)
            all_unique = unique_words(sheet_entries)
            batch_size = compute_batch_size(len(all_unique))

            translation_cache = {}
            processed = 0
            word_bar = st.progress(0)

            for i in range(0, len(all_unique), batch_size):
                batch = all_unique[i : i + batch_size]
                result = translate_batch(batch, llm, rag_store, DOMAIN, FORMALITY, cache=cache)

                for word in batch:
                    translation_cache[word] = result.get(word, translate_single(word, llm, rag_store, DOMAIN, FORMALITY, cache=cache))
                    processed += 1
                    word_bar.progress(min(processed / len(all_unique), 1.0))

            translated_bytes = write_translations(source.data, sheet_entries, translation_cache)
            translated_outputs.append((source, translated_bytes))
            
            for sname, entries in sheet_entries.items():
                for e in entries:
                    all_summary_rows.append({"File": source.name, "English": e.value, "Dutch": translation_cache.get(e.value, "error")})

            overall_bar.progress((file_idx + 1) / len(excel_sources))

        st.success(f"✅ **Done!** Memory updated for {user_name}.")

        # 4. Download
        if len(translated_outputs) == 1:
            source, t_bytes = translated_outputs[0]
            dl_bytes, dl_name = pack_single(t_bytes, source.name)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            dl_bytes, dl_name = pack_zip(translated_outputs)
            mime = "application/zip"

        st.download_button(label=f"⬇️ Download Professional Translation", data=dl_bytes, file_name=dl_name, mime=mime)
        st.dataframe(pd.DataFrame(all_summary_rows).head(30), use_container_width=True)

    except Exception as exc:
        st.error(f"❌ Critical Error: {exc}")
