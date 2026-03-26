"""
app.py — Multi-Model Dutch Business Translator (OpenAI, Claude, Gemini, Local).
Supports Multi-Column Excel and Integrated Dutch Assistant (SENDERUM).
"""

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.config import compute_batch_size, PROVIDERS
from modules.tone_loader import load_tone_from_path
from modules.rag_engine import build_rag_store, RAGStore
from modules.excel_handler import (
    read_word_entries,
    write_translations,
    unique_words,
)
from modules.model_factory import get_chat_model
from modules.translator import translate_batch, translate_single
from modules.chat_engine import DutchAssistant
from modules.cache_manager import CacheManager
from modules.zip_handler import (
    extract_excel_files,
    pack_single,
    pack_zip,
)

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── Session State for Chat ───────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇳🇱 Professional Dutch Hub",
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
.info { background:#eff6ff; border-left:4px solid #3b82f6; border-radius:6px;
        padding:.7rem 1rem; font-size:.87rem; color:#1e40af; margin:.5rem 0 1rem; }
.chat-msg { margin-bottom: 1rem; padding: 0.8rem; border-radius: 10px; }
.user-msg { background: #eef2ff; border-right: 4px solid #4f46e5; text-align: right; }
.ai-msg { background: #f0fdf4; border-left: 4px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:.15rem">
    <span style="font-size:2rem">🇳🇱</span>
    <h1>Dutch Business Hub</h1>
  </div>
  <p class="sub">High-accuracy <strong>SENDERUM</strong> tools for batch files and manual chat.</p>
</div>
""", unsafe_allow_html=True)

# ── Setup ─────────────────────────────────────────────────────────────────────
cache: CacheManager = None
tone_text: str = ""

with st.expander("🛠️ Provider & Identity Setup", expanded=True):
    col_p, col_n = st.columns([1, 1])
    with col_p:
        provider = st.selectbox("Preferred Provider", list(PROVIDERS.keys()), index=0)
    with col_n:
        user_name = st.text_input("User Name", value="default")
        
    is_local = "Local" in provider
    api_key_val = os.getenv(f"{provider.upper().replace(' ','_')}_API_KEY", os.getenv("OPENAI_API_KEY", "")) if "OpenAI" in provider else ""
    
    if is_local:
        api_key = "NOT_NEEDED"
        st.info("✅ **Local Model selected**: Offline CPU mode active.")
    else:
        api_key = st.text_input(f"{provider} API Key", value=api_key_val, type="password")

    if api_key and user_name:
        cache = CacheManager(api_key, user_name)

# ── Global Context (Tone) ─────────────────────────────────────────────────────
TONE_FILE_PATH = "SENDERUM-tone of voice (1).docx"
try:
    tone_text = load_tone_from_path(TONE_FILE_PATH)
except:
    st.error("⚠️ Local tone file not found.")

DOMAIN = "General Business"
FORMALITY = "Formal (u-form)"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_batch, tab_chat = st.tabs(["📊 Batch Translation", "💬 Dutch Assistant"])

# ── TAB 1: Batch Excel ────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="card"><p class="step">Step 1 — Upload files</p>', unsafe_allow_html=True)
    main_upload = st.file_uploader("📊 Excel file or ZIP (Up to 3000+ columns supported)", type=["xlsx", "zip"])
    st.markdown('</div>', unsafe_allow_html=True)

    ready = bool(api_key and user_name and main_upload and tone_text)
    if not ready:
        st.markdown('<div class="warn">🔑 Setup Identity and Upload Excel to continue.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><p class="step">Step 2 — Professional Translation</p>', unsafe_allow_html=True)
    go = st.button("🚀 Start Professional Translation", disabled=not ready)
    st.markdown('</div>', unsafe_allow_html=True)

    if go and ready:
        try:
            llm = get_chat_model(provider, api_key)
            rag_store = RAGStore()
            if not is_local:
                with st.spinner("🔍 Building Tone Index…"):
                    rag_store = build_rag_store(tone_text, provider, api_key)

            excel_sources = extract_excel_files(main_upload)
            overall_bar = st.progress(0)
            translated_outputs = []
            all_summary_rows = []

            for file_idx, source in enumerate(excel_sources):
                st.info(f"Processing `{source.name}`...")
                sheet_entries = read_word_entries(source.data)
                all_unique = unique_words(sheet_entries)
                
                batch_size = 10 if is_local else compute_batch_size(len(all_unique))
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
                overall_bar.progress((file_idx + 1) / len(excel_sources))

            st.success("✅ **Translation Complete!**")
            
            if len(translated_outputs) == 1:
                source, t_bytes = translated_outputs[0]
                dl_bytes, dl_name = pack_single(t_bytes, source.name)
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                dl_bytes, dl_name = pack_zip(translated_outputs)
                mime = "application/zip"

            st.download_button(label=f"⬇️ Download Professional Translation", data=dl_bytes, file_name=dl_name, mime=mime)
        except Exception as exc:
            st.error(f"❌ Error: {exc}")

# ── TAB 2: Dutch Assistant ────────────────────────────────────────────────────
with tab_chat:
    st.markdown('<div class="card"><p class="step">Manual SENDERUM Assistant</p>', unsafe_allow_html=True)
    if not api_key or not user_name:
        st.warning("🔑 Set up your Identity first.")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            role_css = "user-msg" if msg["role"] == "user" else "ai-msg"
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Input
        if prompt := st.chat_input("Ask for a business translation..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Response
            try:
                llm_chat = get_chat_model(provider, api_key)
                rag_chat = build_rag_store(tone_text, provider, api_key) if not is_local else RAGStore()
                assistant = DutchAssistant(llm_chat, rag_chat, DOMAIN, FORMALITY)
                
                with st.chat_message("assistant"):
                    with st.spinner("Translating..."):
                        response = assistant.generate_response(prompt, st.session_state.chat_history)
                        st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Chat setup failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
