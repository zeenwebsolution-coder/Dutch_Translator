"""
app.py — Multi-Model Dutch Business Translator
"""

import os
import streamlit as st
from dotenv import load_dotenv
import traceback
from modules.config import compute_batch_size, PROVIDERS
from modules.tone_loader import load_tone_from_path
from modules.rag_engine import build_rag_store, RAGStore
from modules.excel_handler import (
    read_word_entries as read_excel_entries,
    write_translations as write_excel_translations,
    unique_words as unique_excel_words,
)
from modules.word_handler import (
    read_word_entries as read_docx_entries,
    write_translations as write_docx_translations,
    unique_words as unique_docx_words,
)
from modules.model_factory import get_chat_model
from modules.translator import translate_batch, translate_single
from modules.chat_engine import DutchAssistant
from modules.cache_manager import CacheManager
from modules.zip_handler import extract_excel_files, pack_single, pack_zip

# ── Load ENV ─────────────────────────────────────────
load_dotenv()


# ── Session State ────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="🇳🇱 Dutch Business Hub",
    page_icon="🇳🇱",
    layout="centered",
)

st.title("🇳🇱 Dutch Business Hub")
st.caption("Professional SENDERUM Translation System")

# ── Setup ────────────────────────────────────────────
cache: CacheManager = None
tone_text: str = ""

with st.expander("🛠️ Setup", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    with col2:
        user_name = st.text_input("User Name", value="default")

    is_local = "Local" in provider

    if is_local:
        api_key = "NOT_NEEDED"
        st.info("Local model enabled")
    else:
        api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-proj-...")

    if api_key and user_name:
        cache = CacheManager(api_key, user_name)

# ── Tone File ────────────────────────────────────────
try:
    tone_text = load_tone_from_path("SENDERUM-tone of voice (1).docx")
except Exception:
    st.warning("Tone file not found")

DOMAIN = "General Business"
FORMALITY = "Formal (u-form)"

# ── Tabs ─────────────────────────────────────────────
tab_batch, tab_chat = st.tabs(["📊 Batch", "💬 Assistant"])

# ====================================================
# 📊 BATCH TRANSLATION
# ====================================================
with tab_batch:

    main_upload = st.file_uploader(
        "Upload Excel / Word / ZIP",
        type=["xlsx", "docx", "zip",        
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],        
        help='Supported formats: .xlsx, .docx'
        )

    ready = bool(api_key and user_name and main_upload and tone_text)

    if not ready:
        st.warning("Complete setup first")

    if st.button("🚀 Start Translation", disabled=not ready):

        try:
            llm = get_chat_model(provider, api_key)

            if not is_local:
                with st.spinner("Building tone index..."):
                    rag_store = build_rag_store(tone_text, provider, api_key)
            else:
                rag_store = RAGStore()

            sources = extract_excel_files(main_upload)

            overall_bar = st.progress(0)
            translated_outputs = []

            for idx, source in enumerate(sources):

                st.info(f"Processing {source.name}")
                file_name = source.name.lower()

                # ── Detect type and read entries ──
                if file_name.endswith(".xlsx"):
                    sheet_entries = read_excel_entries(source.data)
                    all_unique = unique_excel_words(sheet_entries)

                elif file_name.endswith(".docx"):
                    # ✅ Pass raw bytes directly — word_handler wraps BytesIO internally
                    sheet_entries = read_docx_entries(source.data)
                    all_unique = unique_docx_words(sheet_entries)

                else:
                    st.error(f"Unsupported file: {source.name}")
                    continue

                # ── Translation ──
                batch_size = 10 if is_local else compute_batch_size(len(all_unique))
                translation_cache = {}
                word_bar = st.progress(0)

                processed = 0

                for i in range(0, len(all_unique), batch_size):
                    batch = all_unique[i:i + batch_size]

                    result = translate_batch(
                        batch, llm, rag_store, DOMAIN, FORMALITY, cache=cache
                    )

                    for word in batch:
                        translation_cache[word] = result.get(
                            word,
                            translate_single(
                                word, llm, rag_store, DOMAIN, FORMALITY, cache=cache
                            )
                        )

                        processed += 1
                        word_bar.progress(processed / len(all_unique))

                # ── Write Output ──
                if file_name.endswith(".xlsx"):
                    translated_bytes = write_excel_translations(
                        source.data, sheet_entries, translation_cache
                    )
                else:
                    # ✅ Pass raw bytes directly — word_handler wraps BytesIO internally
                    translated_bytes = write_docx_translations(
                        source.data, translation_cache
                    )

                translated_outputs.append((source, translated_bytes))
                overall_bar.progress((idx + 1) / len(sources))

            st.success("✅ Translation complete!")

            # ── Download ──
            if len(translated_outputs) == 1:
                source, t_bytes = translated_outputs[0]
                dl_bytes, dl_name = pack_single(t_bytes, source.name)

                if dl_name.endswith(".xlsx"):
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    mime = "application/octet-stream"

            else:
                dl_bytes, dl_name = pack_zip(translated_outputs)
                mime = "application/zip"

            st.download_button("⬇️ Download", dl_bytes, dl_name, mime)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())

# ====================================================
# 💬 CHAT ASSISTANT
# ====================================================
with tab_chat:

    if not api_key or not user_name:
        st.warning("Setup required")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask something..."):

            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("assistant"):
                try:
                    llm = get_chat_model(provider, api_key)

                    rag = (
                        build_rag_store(tone_text, provider, api_key)
                        if not is_local
                        else RAGStore()
                    )

                    assistant = DutchAssistant(llm, rag, DOMAIN, FORMALITY)

                    with st.spinner("Thinking..."):
                        response = assistant.generate_response(
                            prompt, st.session_state.chat_history
                        )
                        st.write(response)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    st.error(f"❌ {e}")