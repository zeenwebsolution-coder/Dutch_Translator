"""
local_translator.py — Local NMT model loader for Helsinki-NLP/opus-mt-en-nl.
Uses Streamlit's cache_resource to keep the model in memory.
"""

import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="📥 Loading Local Dutch Translation Model (300MB)...")
def get_local_pipeline():
    """
    Initialize and cache the Hugging Face translation pipeline.
    This will download the model once and keep it in RAM.
    """
    try:
        from transformers import pipeline
        # Helsinki-NLP/opus-mt-en-nl is compact and highly accurate for NMT.
        model_name = "Helsinki-NLP/opus-mt-en-nl"
        translator = pipeline("translation_en_to_nl", model=model_name)
        return translator
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        return None

def translate_local(pipeline, text: str) -> str:
    """Run a single string through the local translation pipeline."""
    if not pipeline:
        return text
    
    try:
        result = pipeline(text)
        return result[0]['translation_text']
    except Exception as e:
        logger.error(f"Local translation error: {e}")
        return text
