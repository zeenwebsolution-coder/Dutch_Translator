"""
config.py — Central configuration for the Dutch translator.
Updated for Multi-Model support (OpenAI, Anthropic, Google, and Local).
"""

# ── Providers & Models ────────────────────────────────────────────────────────
PROVIDERS = {
    "OpenAI": {
        "chat_model": "gpt-4o",
        "embed_model": "text-embedding-3-small",
    },
    "Anthropic": {
        "chat_model": "claude-3-5-sonnet-20240620",
        "embed_model": None,
    },
    "Google": {
        "chat_model": "gemini-1.5-pro",
        "embed_model": "models/embedding-001",
    },
    "Local (Helsinki-NLP)": {
        "chat_model": "Helsinki-NLP/opus-mt-en-nl",
        "embed_model": None,  # Local model doesn't use RAG in the same way
    }
}

DEFAULT_PROVIDER = "OpenAI"

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
BATCH_THRESHOLDS = [
    (15,   15),
    (50,   20),
    (200,  30),
    (500,  40),
]
DEFAULT_BATCH_SIZE = 50

def compute_batch_size(total_unique_words: int) -> int:
    """Pick the optimal batch size based on how many unique words the file has."""
    for threshold, size in BATCH_THRESHOLDS:
        if total_unique_words <= threshold:
            return min(size, total_unique_words) or 1
    return DEFAULT_BATCH_SIZE

# ── Tone loader ───────────────────────────────────────────────────────────────
MAX_TONE_CHARS      = 8000     # cap raw tone text before chunking

# ── Business domains ──────────────────────────────────────────────────────────
DOMAINS = ["General Business", "Finance & Accounting", "HR & People Management", 
           "Sales & Marketing", "Legal & Compliance", "Operations & Logistics", 
           "IT & Technology", "Customer Service"]

# ── Formality levels ──────────────────────────────────────────────────────────
FORMALITY_OPTIONS = {
    "Formal (u-form)": "Use formal Dutch throughout. Address implied subjects with 'u'.",
    "Semi-formal": "Use semi-formal Dutch. A professional yet approachable register.",
    "Neutral": "Use neutral, standard Dutch (ABN)."
}
