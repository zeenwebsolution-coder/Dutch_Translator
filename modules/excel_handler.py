"""
excel_handler.py — Read and write Excel workbooks.

Responsibilities:
  - Parse every sheet and extract (row, col, value) triples.
  - Write translated values back into the original workbook structure,
    preserving all formatting, styles, and non-word cells.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from openpyxl import load_workbook


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class WordEntry:
    sheet: str
    row: int
    col: int
    value: str


# ── Public API ────────────────────────────────────────────────────────────────

def read_word_entries(file_bytes: bytes) -> dict[str, list[WordEntry]]:
    """
    Open an Excel workbook and extract every non-empty cell value.

    Args:
        file_bytes: Raw bytes of the .xlsx file.

    Returns:
        Dict keyed by sheet name, each value being a list of WordEntry objects.
    """
    wb = load_workbook(io.BytesIO(file_bytes), data_only=True)
    result: dict[str, list[WordEntry]] = {}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        entries: list[WordEntry] = []

        for row in ws.iter_rows():
            for cell in row:
                val = _extract_value(cell.value)
                if val:
                    entries.append(
                        WordEntry(
                            sheet=sheet_name,
                            row=cell.row,
                            col=cell.column,
                            value=val,
                        )
                    )

        if entries:
            result[sheet_name] = entries

    return result


def write_translations(
    original_bytes: bytes,
    sheet_entries: dict[str, list[WordEntry]],
    translation_cache: dict[str, str],
) -> bytes:
    """
    Apply translations to the original workbook (in-memory) and return
    the modified workbook as bytes, preserving all styles and formatting.

    Args:
        original_bytes:    Raw bytes of the source .xlsx file.
        sheet_entries:     Output of read_word_entries() for reference.
        translation_cache: Mapping {english_word: dutch_word}.

    Returns:
        Bytes of the translated .xlsx file.
    """
    wb = load_workbook(io.BytesIO(original_bytes))

    for sheet_name, entries in sheet_entries.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        for entry in entries:
            dutch = translation_cache.get(entry.value, entry.value)
            ws.cell(row=entry.row, column=entry.col).value = dutch

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


def unique_words(sheet_entries: dict[str, list[WordEntry]]) -> list[str]:
    """Return a deduplicated list of all word values across all sheets."""
    seen: set[str] = set()
    result: list[str] = []
    for entries in sheet_entries.values():
        for e in entries:
            if e.value not in seen:
                seen.add(e.value)
                result.append(e.value)
    return result


def total_word_count(sheet_entries: dict[str, list[WordEntry]]) -> int:
    return sum(len(v) for v in sheet_entries.values())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_value(raw) -> str:
    """Coerce a cell value to a clean string, returning '' for empties."""
    if raw is None:
        return ""
    val = str(raw).strip()
    return "" if val.lower() in ("none", "nan", "") else val
