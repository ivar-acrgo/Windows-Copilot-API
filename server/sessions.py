"""In-memory mapping from OpenAI message history to Copilot conversation ids.

Stateless clients (e.g. Cherry Studio) resend the full ``messages`` array each
turn but never pass ``conversation_id``. We derive a stable key from the user
turns seen so far, look up the Copilot id, and only forward the latest user
message upstream — matching the web client's multi-turn behaviour.
"""

import threading
import time
from typing import Optional


class ConversationSessionStore:
    """Thread-safe ``session_key -> copilot_conversation_id`` cache with TTL."""

    def __init__(self, ttl_seconds: float):
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: Optional[str]) -> Optional[str]:
        if not key or not self.enabled:
            return None
        with self._lock:
            self._evict_expired()
            entry = self._entries.get(key)
            if entry is None:
                return None
            conv_id, expires = entry
            if time.monotonic() > expires:
                del self._entries[key]
                return None
            return conv_id

    def put(self, key: Optional[str], conversation_id: Optional[str]) -> None:
        if not key or not conversation_id or not self.enabled:
            return
        with self._lock:
            self._entries[key] = (conversation_id, time.monotonic() + self._ttl)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        for key, (_, expires) in list(self._entries.items()):
            if now > expires:
                del self._entries[key]
