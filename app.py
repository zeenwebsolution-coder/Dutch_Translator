"""
app.py — Streamlit UI for the Dutch Business Translator.

Orchestrates all modules:
  tone_loader   → load & clean Dutch reference text
  rag_engine    → LangChain FAISS vector store for tone retrieval
  excel_handler → parse words from Excel, write translations back
  translator    → GPT-4o-mini calls (smart dynamic batch sizing)
  zip_handler   → unpack zip uploads, repack translated files into zip
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
.stButton>button {
    background:#2563eb; color:white; border:none; border-radius:8px;
    padding:.6rem 2rem; font-weight:600; font-size:1rem; width:100%;
    margin-top:.4rem; transition:background .2s;
}
.stButton>button:hover { background:#1d4ed8; }
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
    Upload a single Excel file <em>or a ZIP of multiple Excel files</em>.<br>
    LangChain RAG retrieves the most relevant tone examples per batch —
    GPT-4o-mini translates with smart auto-sized batches for optimal accuracy.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Step 0: API key ───────────────────────────────────────────────────────────
with st.expander("🔑 OpenAI API Key", expanded=not bool(env_api_key)):
    api_key = st.text_input(
        "Enter your OpenAI API key",
        value=env_api_key,
        type="password",
        placeholder="sk-…",
    )
    if env_api_key:
        st.caption("✅ Pre-loaded from .env file. You can override it above.")
    else:
        st.caption("Used only for this session. Never stored or logged.")

# ── Step 1: File uploads ──────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 1 — Upload files</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    main_upload = st.file_uploader(
        "📊 Excel file(s)",
        type=["xlsx", "zip"],
        help=(
            "Single .xlsx  →  returns one translated .xlsx\n"
            ".zip of .xlsx files  →  returns a translated .zip\n"
            "Nested folders inside zip are supported."
        ),
    )
with col2:
    tone_file = st.file_uploader(
        "📄 Dutch tone reference (optional)",
        type=["doc", "docx", "txt", "rtf", "csv", "xlsx", "xls", "pdf"],
        help=(
            "Any Dutch business document that defines the tone to match.\n"
            "Accepted: .doc  .docx  .txt  .rtf  .csv  .xlsx  .xls  .pdf"
        ),
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
    '<div class="info">⚡ <strong>Smart batching</strong> — batch size is automatically '
    'optimized per file based on the number of unique words. Smaller files use larger '
    'batches (fewer API calls), larger files use efficient chunking.</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Parse & preview uploaded files ───────────────────────────────────────────
excel_sources = []
tone_text     = ""

if main_upload:
    try:
        excel_sources = extract_excel_files(main_upload)
        pills = "".join(
            f'<span class="file-pill">📄 {s.name}</span>'
            for s in excel_sources
        )
        st.markdown(
            f'<div class="ok">✅ {len(excel_sources)} Excel file(s) ready:<br>{pills}</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"Preview: {excel_sources[0].name}", expanded=True):
            first_entries = read_word_entries(excel_sources[0].data)
            tabs = st.tabs(list(first_entries.keys())[:6])
            for tab, sname in zip(tabs, list(first_entries.keys())[:6]):
                with tab:
                    entries = first_entries[sname]
                    st.caption(f"{len(entries)} word(s)")
                    for e in entries[:6]:
                        st.markdown(f"- {e.value}")
                    if len(entries) > 6:
                        st.caption(f"… and {len(entries)-6} more")

    except ZipHandlerError as exc:
        st.error(str(exc))
        excel_sources = []

if tone_file:
    try:
        tone_text = load_tone_file(tone_file)
        st.markdown(
            f'<div class="info">📚 Tone file loaded — '
            f'{len(tone_text):,} chars ready for LangChain RAG indexing.</div>',
            unsafe_allow_html=True,
        )
    except ToneLoadError as exc:
        st.error(str(exc))

# ── Step 3: Translate ─────────────────────────────────────────────────────────
st.markdown('<div class="card"><p class="step">Step 3 — Translate</p>',
            unsafe_allow_html=True)

ready = bool(api_key and excel_sources)
if not ready:
    st.markdown(
        '<div class="warn">⚠️ Provide your API key and upload at least one '
        'Excel file or ZIP to continue.</div>',
        unsafe_allow_html=True,
    )

go = st.button("🚀 Translate all files to Dutch", disabled=not ready)
st.markdown('</div>', unsafe_allow_html=True)

# ── Translation pipeline ──────────────────────────────────────────────────────
if go and ready:
    client = OpenAI(api_key=api_key)

    # ── Validate API key first ────────────────────────────────────────────
    with st.spinner("🔑 Validating API key…"):
        try:
            client.models.list()
        except Exception as exc:
            st.error(
                f"❌ **API Key Error**: Could not connect to OpenAI. "
                f"Please check your API key.\n\nDetails: {exc}"
            )
            st.stop()
    st.success("🔑 API key validated successfully.")

    # ── Phase 1: Build RAG store once, shared across all files ────────────
    rag_store: RAGStore = RAGStore()
    if tone_text:
        with st.spinner("🔍 Building LangChain RAG index from tone file…"):
            try:
                rag_store = build_rag_store(tone_text, api_key)
                st.success(
                    f"RAG index ready: {rag_store.chunk_count} tone chunks "
                    f"embedded via LangChain FAISS."
                )
            except Exception as exc:
                st.error(f"❌ Failed to build RAG index: {exc}")
                st.warning("Continuing without tone context…")
    else:
        st.info("No tone file — using domain + formality context only.")

    # ── Phase 2: Translate each file (file-by-file with smart batching) ──
    translated_outputs: list[tuple] = []
    all_summary_rows:   list[dict]  = []

    overall_bar    = st.progress(0)
    overall_status = st.empty()

    for file_idx, source in enumerate(excel_sources):
        overall_status.markdown(
            f"**File {file_idx+1} / {len(excel_sources)} — `{source.name}`**"
        )

        sheet_entries = read_word_entries(source.data)
        all_unique    = unique_words(sheet_entries)

        # ── Smart batch sizing for this file ──────────────────────────
        batch_size = compute_batch_size(len(all_unique))
        st.markdown(
            f'<div class="batch-info">📐 <strong>{source.name}</strong>: '
            f'{len(all_unique)} unique words → batch size = {batch_size}</div>',
            unsafe_allow_html=True,
        )

        word_bar    = st.progress(0)
        word_status = st.empty()

        translation_cache: dict[str, str] = {}
        processed = 0

        for i in range(0, len(all_unique), batch_size):
            batch = all_unique[i : i + batch_size]
            batch_end = min(i + batch_size, len(all_unique))
            word_status.markdown(
                f"  ↳ translating words {i+1}–{batch_end} "
                f"of {len(all_unique)}…"
            )

            result = translate_batch(batch, client, rag_store, domain, formality)

            for word in batch:
                translation_cache[word] = (
                    result[word] if word in result
                    else translate_single(word, client, rag_store, domain, formality)
                )
                processed += 1
                word_bar.progress(min(processed / len(all_unique), 1.0))

        word_status.markdown(
            f"  ✅ `{source.name}` done — {len(all_unique)} unique words "
            f"(batch size: {batch_size})."
        )

        translated_bytes = write_translations(
            source.data, sheet_entries, translation_cache
        )
        translated_outputs.append((source, translated_bytes))

        for sname, entries in sheet_entries.items():
            for e in entries:
                all_summary_rows.append({
                    "File":               source.name,
                    "Sheet":              sname,
                    "English":            e.value,
                    "Dutch (translated)": translation_cache.get(e.value, e.value),
                })

        overall_bar.progress((file_idx + 1) / len(excel_sources))

    overall_status.markdown(
        f"✅ **All {len(excel_sources)} file(s) translated successfully!**"
    )

    # ── Phase 3: Package & download ───────────────────────────────────────
    if len(translated_outputs) == 1:
        source, t_bytes = translated_outputs[0]
        dl_bytes, dl_name = pack_single(t_bytes, source.name)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        dl_bytes, dl_name = pack_zip(translated_outputs)
        mime = "application/zip"

    st.markdown("### 📋 Full translation summary")
    st.dataframe(
        pd.DataFrame(all_summary_rows), use_container_width=True, height=340
    )

    st.download_button(
        label=f"⬇️ Download  {dl_name}",
        data=dl_bytes,
        file_name=dl_name,
        mime=mime,
    )
    st.success(
        f"🎉 {len(all_summary_rows)} translations across "
        f"{len(excel_sources)} file(s). Download **{dl_name}** above."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:2rem;border:none;border-top:1px solid #e5e7eb">
<p style="text-align:center;color:#9ca3af;font-size:.8rem">
  Powered by GPT-4o-mini + LangChain RAG (FAISS) · Smart auto-batching · Dutch business localisation
</p>
""", unsafe_allow_html=True)
