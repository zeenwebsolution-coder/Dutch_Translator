"""
app.py — Streamlit UI for the Dutch Business Translator.

Targeted Excel logic + Persistent User-wise Cache (SQLite).
Automatically finds "English" columns and writes to "Dutch" columns next to them.
"""

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from modules.config import DOMAINS, FORMALITY_OPTIONS, compute_batch_size
from modules.tone_loader import load_tone_file, ToneLoadError
from modules.rag_engine import build_rag_store, RAGStore
from modules.excel_handler import (
    read_word_entries,
    write_translations,
    unique_words,
    total_word_count,
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
    page_title="EN → NL Business Translator",
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
.sub { color:#6b7280; font-size:.92rem; margin:.25rem 0 1.2rem; }
.step { font-size:.75rem; font-weight:700; letter-spacing:.09em;
        color:#2563eb; text-transform:uppercase; margin-bottom:.35rem; }
.info { background:#eff6ff; border-left:4px solid #3b82f6; border-radius:6px;
        padding:.7rem 1rem; font-size:.87rem; color:#1e40af; margin:.5rem 0 1rem; }
.warn { background:#fff7ed; border-left:4px solid #f59e0b; border-radius:6px;
        padding:.7rem 1rem; font-size:.87rem; color:#92400e; margin:.5rem 0; }
.ok   { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px;
        padding:.7rem 1rem; font-size:.87rem; color:#166534; margin:.5rem 0 1rem; }
.file-pill {
    display:inline-block; background:#eff6ff; color:#1e40af;
    border-radius:20px; padding:.25rem .85rem; font-size:.82rem;
    margin:.2rem .2rem; font-weight:500;
}
.batch-info {
    display:inline-block; background:#f0fdf4; color:#166534;
    border-radius:8px; padding:.35rem .85rem; font-size:.82rem;
    margin:.3rem 0; font-weight:500; border: 1px solid #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:.15rem">
    <span style="font-size:2rem">🇳🇱</span>
    <h1>English → Dutch Business Translator</h1>
  </div>
  <p class="sub">
    <strong>Improved!</strong> Targeting the "English" column specifically and writing to the "Dutch" column next to it.<br>
    Includes <strong>Persistent Cache</strong> — translations stay intact on Streamlit Cloud (user-wise).
  </p>
</div>
""", unsafe_allow_html=True)

# ── Step 0: API key & Cache ───────────────────────────────────────────────────
cache: CacheManager = None

with st.expander("🔑 Setup & Persistent Cache Memory", expanded=not bool(env_api_key)):
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        value=env_api_key,
        type="password",
        placeholder="sk-…",
    )
    if api_key_input:
        cache = CacheManager(api_key_input)
        st.markdown(f"💾 **Cache Storage**: {cache.get_stats()} translations remembered for your key.")
        if st.button("🧹 Clear User Cache"):
            cache.clear()
            st.success("User cache cleared!")
    else:
        st.caption("Enter your API key to activate persistent translation memory.")

# ── Step 1: File uploads ──────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 1 — Upload files</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    main_upload = st.file_uploader(
        "📊 Excel file(s) (Targeting 'English' column)",
        type=["xlsx", "zip"],
    )
with col2:
    tone_file = st.file_uploader(
        "📄 Dutch tone reference (optional)",
        type=["doc", "docx", "txt", "rtf", "csv", "xlsx", "xls", "pdf"],
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Step 2: Options ───────────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 2 — Options</p>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    domain = st.selectbox("Business domain", DOMAINS)
with col_b:
    formality = st.selectbox("Formality level", list(FORMALITY_OPTIONS.keys()))

st.markdown(
    '<div class="info">💡 <strong>Excel Tip</strong>: If your sheets have columns labeled "English" or "EN", '
    'the app will find them automatically and write results into the "Dutch" column next to them.</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Parse & preview ───────────────────────────────────────────────────────────
excel_sources = []
tone_text     = ""

if main_upload:
    try:
        excel_sources = extract_excel_files(main_upload)
        pills = "".join(f'<span class="file-pill">📄 {s.name}</span>' for s in excel_sources)
        st.markdown(f'<div class="ok">✅ {len(excel_sources)} file(s) ready:<br>{pills}</div>', unsafe_allow_html=True)
    except ZipHandlerError as exc:
        st.error(str(exc))

if tone_file:
    try:
        tone_text = load_tone_file(tone_file)
        st.info(f"📚 Tone file loaded ({len(tone_text)//1024} KB).")
    except ToneLoadError as exc:
        st.error(str(exc))

# ── Step 3: Translate ─────────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 3 — Translate</p>',
            unsafe_allow_html=True)

ready = bool(api_key_input and excel_sources)
if not ready:
    st.markdown('<div class="warn">⚠️ API key and Excel file required.</div>', unsafe_allow_html=True)

go = st.button("🚀 Process Column-by-Column & Save to Cache", disabled=not ready)
st.markdown('</div>', unsafe_allow_html=True)

# ── Translation pipeline ──────────────────────────────────────────────────────
if go and ready:
    client = OpenAI(api_key=api_key_input)

    # 1. Build RAG Store
    rag_store: RAGStore = RAGStore()
    if tone_text:
        with st.spinner("🔍 Building RAG index…"):
            try:
                rag_store = build_rag_store(tone_text, api_key_input)
                st.success(f"RAG ready: {rag_store.chunk_count} chunks embedded.")
            except Exception as e:
                st.error(f"RAG error: {e}")

    # 2. Translate File-by-File
    translated_outputs: list[tuple] = []
    all_summary_rows:   list[dict]  = []

    overall_bar    = st.progress(0)
    for file_idx, source in enumerate(excel_sources):
        st.markdown(f"---")
        st.info(f"**Currently Processing: `{source.name}`**")

        sheet_entries = read_word_entries(source.data)
        all_unique    = unique_words(sheet_entries)
        batch_size    = compute_batch_size(len(all_unique))

        st.markdown(f'<div class="batch-info">Sheet Word Count: {len(all_unique)} | Smart Batch Size: {batch_size}</div>', unsafe_allow_html=True)

        translation_cache: dict[str, str] = {}
        processed = 0
        word_bar    = st.progress(0)

        for i in range(0, len(all_unique), batch_size):
            batch = all_unique[i : i + batch_size]
            
            # Use persistent cache if available
            result = translate_batch(batch, client, rag_store, domain, formality, cache=cache)

            for word in batch:
                translation_cache[word] = result.get(word, 
                                          translate_single(word, client, rag_store, domain, formality, cache=cache))
                processed += 1
                word_bar.progress(min(processed / len(all_unique), 1.0))

        # 3. Write specifically back to Dutch Column
        translated_bytes = write_translations(source.data, sheet_entries, translation_cache)
        translated_outputs.append((source, translated_bytes))

        # Summary for preview
        for sname, entries in sheet_entries.items():
            for e in entries:
                all_summary_rows.append({
                    "File": source.name,
                    "English": e.value,
                    "Dutch (translated)": translation_cache.get(e.value, "error"),
                    "Target Column": f"Col {e.target_col}"
                })

        overall_bar.progress((file_idx + 1) / len(excel_sources))

    st.success("✅ **All processing completed!** Memory has been updated.")

    # 4. Final Download
    if len(translated_outputs) == 1:
        source, t_bytes = translated_outputs[0]
        dl_bytes, dl_name = pack_single(t_bytes, source.name)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        dl_bytes, dl_name = pack_zip(translated_outputs)
        mime = "application/zip"

    st.download_button(label=f"⬇️ Download Translated File(s)", data=dl_bytes, file_name=dl_name, mime=mime)
    st.dataframe(pd.DataFrame(all_summary_rows).head(50), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""<hr><p style="text-align:center;color:#9ca3af;font-size:.8rem">
  Smart Excel Handler + Persistent Cache Memory enabled.</p>""", unsafe_allow_html=True)
