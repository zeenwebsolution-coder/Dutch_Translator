"""
excel_handler.py — Targeted reading and writing for Excel workbooks.

Responsibilities:
  - Find "English" and "Dutch" columns in each sheet.
  - Extract only text from the source column.
  - Write translations back to the targeted Dutch column (next to English).
"""

from __future__ import annotations

import io
import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from openpyxl import load_workbook
from openpyxl.cell import Cell, MergedCell

logger = logging.getLogger(__name__)

# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class WordEntry:
    sheet: str
    row: int
    col: int
    target_col: int
    value: str

# ── Header Matching ───────────────────────────────────────────────────────────

# Match any column containing English, EN, Source, or Original (even with extra text)
SOURCE_HEADER_REGEX = re.compile(r".*(english|en|source|original).*", re.IGNORECASE)
# Match any column containing Dutch, NL, Target, or Translation (even with extra text)
TARGET_HEADER_REGEX = re.compile(r".*(dutch|nl|target|vertaling|translation).*", re.IGNORECASE)

def _find_columns(ws) -> Tuple[Optional[int], Optional[int], int]:
    """
    Search first 10 rows for column indices of Source and Target.
    Return (source_col, target_col, header_row_index).
    """
    source_col = None
    target_col = None
    header_row_idx = 0

    # Look through first 10 rows to find headers (expanded from 5)
    for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=10), start=1):
        for cell in row:
            val = str(cell.value).strip() if cell.value else ""
            if not val:
                continue
            
            if SOURCE_HEADER_REGEX.match(val):
                source_col = cell.column
                header_row_idx = row_idx
            elif TARGET_HEADER_REGEX.match(val):
                target_col = cell.column
                header_row_idx = row_idx
        
        # If we found both, we can stop the loop early
        if source_col and target_col:
            break
    
    # Fallback: if no source found but sheet has data, assume Col 1 is source
    if source_col is None:
        source_col = 1
        header_row_idx = 1
    
    # If no target found, use column next to source
    if target_col is None:
        target_col = source_col + 1
        
    return source_col, target_col, header_row_idx

# ── Public API ────────────────────────────────────────────────────────────────

def read_word_entries(file_bytes: bytes) -> dict[str, list[WordEntry]]:
    """
    Open an Excel workbook and extract words ONLY from the source English column.
    """
    wb = load_workbook(io.BytesIO(file_bytes), data_only=True)
    result: dict[str, list[WordEntry]] = {}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        source_col, target_col, header_row = _find_columns(ws)
        
        entries: list[WordEntry] = []
        
        # Iterate only through the source column, starting below header
        for row in ws.iter_rows(min_row=header_row + 1):
            cell = row[source_col - 1]
            val = _extract_value(cell.value)
            if val:
                entries.append(
                    WordEntry(
                        sheet=sheet_name,
                        row=cell.row,
                        col=cell.column,
                        target_col=target_col,
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
    Apply translations specifically to the targeted Dutch column.
    Safely handles MergedCells which are read-only.
    """
    wb = load_workbook(io.BytesIO(original_bytes))

    for sheet_name, entries in sheet_entries.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        
        for entry in entries:
            dutch = translation_cache.get(entry.value, entry.value)
            
            # Access the cell object
            cell = ws.cell(row=entry.row, column=entry.target_col)
            
            # Check if it's a MergedCell (which is read-only)
            if isinstance(cell, MergedCell):
                logger.debug(f"Skipping MergedCell at {sheet_name} R{entry.row} C{entry.target_col}")
                continue
                
            cell.value = dutch

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
