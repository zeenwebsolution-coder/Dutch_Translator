"""
tone_loader.py — Parse the Dutch tone reference file (Local or Uploaded).
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import pandas as pd
from modules.config import MAX_TONE_CHARS

class ToneLoadError(Exception):
    """Raised when the tone file cannot be parsed."""

SUPPORTED_EXTENSIONS = {
    ".doc", ".docx", ".txt", ".rtf",
    ".csv", ".xlsx", ".xls", ".pdf",
}

def load_tone_from_path(file_path: str) -> str:
    """Read a local file from disk and return its cleaned Dutch text."""
    if not os.path.exists(file_path):
        raise ToneLoadError(f"Tone reference file not found at: {file_path}")
    
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    
    name = os.path.basename(file_path).lower()
    return _parse_raw_content(name, raw_bytes)

def load_tone_file(uploaded_file) -> str:
    """Read an uploaded file and return cleaned Dutch text."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return _parse_raw_content(name, raw_bytes)

def _parse_raw_content(name: str, raw_bytes: bytes) -> str:
    """Internal helper to dispatch to correct format handler."""
    ext = _get_ext(name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ToneLoadError(f"Unsupported format: '{name}'")

    try:
        if ext == ".txt":
            raw = _load_txt(raw_bytes)
        elif ext == ".docx":
            raw = _load_docx(raw_bytes)
        elif ext == ".pdf":
            raw = _load_pdf(raw_bytes)
        else:
            # Fallback for local files to try .docx or similar
            raw = _load_docx(raw_bytes)
    except Exception as exc:
        raise ToneLoadError(f"Could not read '{name}': {exc}")

    cleaned = _clean(raw)
    if not cleaned:
        raise ToneLoadError(f"No text extracted from '{name}'.")
    return cleaned[:MAX_TONE_CHARS]

# ── Format handlers ───────────────────

def _load_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError: continue
    return data.decode("utf-8", errors="ignore")

def _load_docx(data: bytes) -> str:
    import io
    from docx import Document
    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip(): paragraphs.append(cell.text.strip())
    return " ".join(paragraphs)

def _load_pdf(data: bytes) -> str:
    import io
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    out = io.StringIO()
    extract_text_to_fp(io.BytesIO(data), out, laparams=LAParams(), output_type="text", codec="utf-8")
    return out.getvalue()

def _get_ext(filename_lower: str) -> str:
    for ext in (".docx", ".doc", ".xlsx", ".xls", ".txt", ".csv", ".pdf", ".rtf"):
        if filename_lower.endswith(ext): return ext
    return ""

def _clean(text: str) -> str:
    text = re.sub(r"[\r\n\t\x0b\x0c]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[\x00-\x08\x0e-\x1f\x7f]", "", text)
    return text.strip()
