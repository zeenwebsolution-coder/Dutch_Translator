"""
zip_handler.py — Handle bulk Excel file input and output.

Supports:
  - Single .xlsx file
  - .zip archive containing any number of .xlsx files (nested folders OK)

Returns:
  - Single translated .xlsx   if input was a single file
  - .zip of translated files  if input was a zip archive
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass
class ExcelSource:
    """One Excel file extracted from the upload."""
    name: str          # original filename (no path)
    arc_path: str      # path inside the zip (empty for single-file uploads)
    data: bytes        # raw .xlsx bytes


class ZipHandlerError(Exception):
    pass


# ── Input ─────────────────────────────────────────────────────────────────────

def extract_excel_files(uploaded_file) -> list[ExcelSource]:
    """
    Given a Streamlit UploadedFile, return a list of ExcelSource objects.

    Accepts:
      - A single .xlsx file  → returns [ExcelSource]
      - A .zip file          → returns [ExcelSource, ...] for every .xlsx inside

    Raises:
        ZipHandlerError: if the zip is invalid or contains no .xlsx files.
    """
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".xlsx"):
        return [ExcelSource(
            name=uploaded_file.name,
            arc_path="",
            data=raw,
        )]

    if name.endswith(".zip"):
        return _extract_from_zip(raw)

    raise ZipHandlerError(
        f"Unsupported upload format: '{uploaded_file.name}'. "
        "Please upload a .xlsx file or a .zip archive of .xlsx files."
    )


def _extract_from_zip(raw: bytes) -> list[ExcelSource]:
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise ZipHandlerError("The uploaded .zip file appears to be corrupted.")

    sources: list[ExcelSource] = []
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        for entry in zf.infolist():
            # Skip macOS metadata files and directories
            if entry.filename.startswith("__MACOSX") or entry.is_dir():
                continue
            p = PurePosixPath(entry.filename)
            if p.suffix.lower() != ".xlsx":
                continue
            data = zf.read(entry.filename)
            sources.append(ExcelSource(
                name=p.name,
                arc_path=entry.filename,
                data=data,
            ))

    if not sources:
        raise ZipHandlerError(
            "No .xlsx files found inside the uploaded zip archive. "
            "Make sure the zip contains Excel files."
        )

    return sources


# ── Output ────────────────────────────────────────────────────────────────────

def pack_single(translated_bytes: bytes, original_name: str) -> tuple[bytes, str]:
    """Return (bytes, filename) for a single translated file."""
    out_name = _nl_name(original_name)
    return translated_bytes, out_name


def pack_zip(translated: list[tuple[ExcelSource, bytes]]) -> tuple[bytes, str]:
    """
    Pack all translated Excel files back into a zip archive.

    Args:
        translated: List of (ExcelSource, translated_bytes) pairs.

    Returns:
        (zip_bytes, zip_filename)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, data in translated:
            # Preserve original subfolder structure inside the zip
            if source.arc_path:
                folder = str(PurePosixPath(source.arc_path).parent)
                out_path = (
                    f"{folder}/{_nl_name(source.name)}"
                    if folder != "."
                    else _nl_name(source.name)
                )
            else:
                out_path = _nl_name(source.name)
            zf.writestr(out_path, data)

    buf.seek(0)
    return buf.read(), "translated_NL.zip"


def _nl_name(filename: str) -> str:
    """Append _NL before the .xlsx extension."""
    if filename.lower().endswith(".xlsx"):
        return filename[:-5] + "_NL.xlsx"
    return filename + "_NL.xlsx"
