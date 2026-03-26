"""
cache_manager.py — Persistent SQLite-based cache for translations.
Segregates cache by API key hash AND user name for multi-user support.
"""

import sqlite3
import hashlib
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache database path (local filesystem)
CACHE_DB_PATH = "translations_cache.db"

class CacheManager:
    def __init__(self, api_key: str, user_name: str = "default"):
        """Initialize the cache for a specific user."""
        # Create a unique session ID based on BOTH API key and User Name
        raw_id = f"{api_key}_{user_name.strip().lower()}"
        self.session_id = hashlib.sha256(raw_id.encode()).hexdigest()
        self.user_name = user_name.strip()
        self._init_db()

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    key_hash TEXT,
                    domain TEXT,
                    formality TEXT,
                    english_text TEXT,
                    dutch_text TEXT,
                    PRIMARY KEY (key_hash, domain, formality, english_text)
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize cache DB: {e}")

    def get(self, domain: str, formality: str, english_text: str) -> Optional[str]:
        """Retrieve a translation from the cache."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                SELECT dutch_text FROM translations 
                WHERE key_hash=? AND domain=? AND formality=? AND english_text=?
            """, (self.session_id, domain, formality, english_text.strip()))
            result = cur.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    def set(self, domain: str, formality: str, english_text: str, dutch_text: str):
        """Store a translation in the cache."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO translations 
                (key_hash, domain, formality, english_text, dutch_text) 
                VALUES (?, ?, ?, ?, ?)
            """, (self.session_id, domain, formality, english_text.strip(), dutch_text.strip()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache set failed: {e}")

    def clear(self):
        """Clear the cache for this specific user session."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute("DELETE FROM translations WHERE key_hash=?", (self.session_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    def get_stats(self):
        """Return number of cached entries for this user session."""
        try:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM translations WHERE key_hash=?", (self.session_id,))
            count = cur.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
