"""Turn OpenAI ``messages`` into a single Copilot prompt per request.

Copilot's protocol has no role/system channel — it takes one prompt string per
turn. For multi-turn chats we send only the latest user message (plus any
system prefix) and rely on Copilot's server-side history via ``conversation_id``.
"""

import hashlib
import json
from typing import Any, List, Optional, Union

from .schemas import ChatMessage


def content_text(content: Optional[Union[str, List[Any]]]) -> str:
    """Extract plain text from a message's content (string or content-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
        else:
            parts.append(str(part))
    return "\n".join(p for p in parts if p)


def _user_texts(messages: List[ChatMessage]) -> List[str]:
    return [
        content_text(m.content)
        for m in messages
        if m.role == "user" and content_text(m.content).strip()
    ]


def _hash_user_texts(user_texts: List[str]) -> str:
    return hashlib.sha256(
        json.dumps(user_texts, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def messages_session_key(messages: List[ChatMessage]) -> Optional[str]:
    """Lookup key for an in-flight multi-turn chat, or ``None`` for a new thread.

    Keys are derived from the sequence of *user* messages only (all but the
    latest), so assistant wording drift in the client history does not break
    continuation.
    """
    users = _user_texts(messages)
    if len(users) <= 1:
        return None
    return _hash_user_texts(users[:-1])


def messages_store_key(messages: List[ChatMessage]) -> Optional[str]:
    """Key under which to cache ``conversation_id`` after a successful turn."""
    users = _user_texts(messages)
    if not users:
        return None
    return _hash_user_texts(users)


def turn_prompt(messages: List[ChatMessage]) -> str:
    """Build the Copilot prompt for one turn: optional system prefix + last user."""
    system = "\n\n".join(
        content_text(m.content) for m in messages if m.role == "system" and m.content
    )
    users = _user_texts(messages)
    body = users[-1] if users else ""
    if system and body:
        return f"{system}\n\n{body}"
    return system or body


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten an OpenAI ``messages`` array into a single Copilot prompt.

    Deprecated for the HTTP server path — kept for callers that intentionally
    want the old all-in-one transcript shape.
    """
    system = "\n\n".join(
        content_text(m.content) for m in messages if m.role == "system" and m.content
    )
    convo = [m for m in messages if m.role != "system"]

    if len(convo) == 1 and convo[0].role == "user":
        body = content_text(convo[0].content)
    else:
        lines = []
        for m in convo:
            label = "User" if m.role == "user" else "Assistant"
            lines.append(f"{label}: {content_text(m.content)}")
        lines.append("Assistant:")
        body = "\n".join(lines)

    if system and body:
        return f"{system}\n\n{body}"
    return system or body
