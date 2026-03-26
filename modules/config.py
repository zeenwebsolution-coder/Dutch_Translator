"""
config.py — Central configuration for the Dutch translator.
All constants, model names, and tunable settings live here.
"""

# ── Models ────────────────────────────────────────────────────────────────────
TRANSLATION_MODEL   = "gpt-4o-mini"
EMBEDDING_MODEL     = "text-embedding-3-small"   # used by LangChain OpenAIEmbeddings

# ── RAG settings ──────────────────────────────────────────────────────────────
CHUNK_SIZE          = 200      # characters per tone chunk
CHUNK_OVERLAP       = 40       # overlap between adjacent chunks
TOP_K_CHUNKS        = 5        # number of tone chunks retrieved per batch
MIN_CHUNK_LENGTH    = 15       # discard chunks shorter than this

# ── Translation settings ──────────────────────────────────────────────────────
MAX_RETRIES         = 3        # retry attempts on API failure
TEMPERATURE         = 0.1      # near-deterministic — accuracy first
MAX_TOKENS          = 2000     # sufficient headroom for larger batches

# ── Smart batch sizing ───────────────────────────────────────────────────────
# Batch size is determined dynamically based on total unique words in a file.
# gpt-4o-mini handles larger batches well — no need for conservative fixed 10.
BATCH_THRESHOLDS = [
    # (max_words, batch_size)
    (15,   15),     # tiny files: send all at once
    (50,   20),     # small files: batches of 20
    (200,  30),     # medium files: batches of 30
    (500,  40),     # large files: batches of 40
]
DEFAULT_BATCH_SIZE = 50  # very large files: batches of 50


def compute_batch_size(total_unique_words: int) -> int:
    """Pick the optimal batch size based on how many unique words the file has."""
    for threshold, size in BATCH_THRESHOLDS:
        if total_unique_words <= threshold:
            return min(size, total_unique_words) or 1
    return DEFAULT_BATCH_SIZE


# ── Tone loader ───────────────────────────────────────────────────────────────
MAX_TONE_CHARS      = 8000     # cap raw tone text before chunking

# ── Business domains ──────────────────────────────────────────────────────────
DOMAINS = [
    "General Business",
    "Finance & Accounting",
    "HR & People Management",
    "Sales & Marketing",
    "Legal & Compliance",
    "Operations & Logistics",
    "IT & Technology",
    "Customer Service",
]

# ── Formality levels ──────────────────────────────────────────────────────────
FORMALITY_OPTIONS = {
    "Formal (u-form)": (
        "Use formal Dutch throughout. Address implied subjects with 'u'. "
        "Prefer formal, professional terminology as used in official business documents."
    ),
    "Semi-formal": (
        "Use semi-formal Dutch. A professional yet approachable register, "
        "suitable for internal business communication."
    ),
    "Neutral": (
        "Use neutral, standard Dutch (ABN — Algemeen Beschaafd Nederlands). "
        "Avoid overly formal or colloquial expressions."
    ),
}
