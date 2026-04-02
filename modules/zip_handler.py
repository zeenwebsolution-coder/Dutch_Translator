from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass
class ExcelSource:
    name: str
    arc_path: str
    data: bytes


class ZipHandlerError(Exception):
    pass


# ── Input ─────────────────────────────────────────────

def extract_excel_files(uploaded_file) -> list[ExcelSource]:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".xlsx") or name.endswith(".docx"):
        return [ExcelSource(
            name=uploaded_file.name,
            arc_path="",
            data=raw,
        )]

    if name.endswith(".zip"):
        return _extract_from_zip(raw)

    raise ZipHandlerError(
        f"Unsupported format: '{uploaded_file.name}'. "
        "Please upload a .xlsx, .docx file or a .zip archive of .xlsx/.docx files."
    )


def _extract_from_zip(raw: bytes) -> list[ExcelSource]:
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise ZipHandlerError("Invalid zip file")

    sources = []

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        for entry in zf.infolist():

            if entry.filename.startswith("__MACOSX") or entry.is_dir():
                continue

            p = PurePosixPath(entry.filename)

            # ✅ FIXED condition
            if p.suffix.lower() not in [".xlsx", ".docx"]:
                continue

            data = zf.read(entry.filename)

            sources.append(ExcelSource(
                name=p.name,
                arc_path=entry.filename,
                data=data,
            ))

    if not sources:
        raise ZipHandlerError("No valid files found in zip")

    return sources


# ── Output ────────────────────────────────────────────

def pack_single(translated_bytes: bytes, original_name: str) -> tuple[bytes, str]:
    return translated_bytes, _nl_name(original_name)


def pack_zip(translated: list[tuple[ExcelSource, bytes]]) -> tuple[bytes, str]:
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, data in translated:

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
    if filename.lower().endswith(".xlsx"):
        return filename[:-5] + "_NL.xlsx"
    elif filename.lower().endswith(".docx"):
        return filename[:-5] + "_NL.docx"
    return filename + "_NL"