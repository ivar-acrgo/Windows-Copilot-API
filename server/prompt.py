"""Turn OpenAI ``messages`` into a single Copilot prompt per request.

Copilot's protocol has no role/system channel — it takes one prompt string per
turn. For multi-turn chats we send only the latest user message (plus any
system prefix) and rely on Copilot's server-side history via ``conversation_id``.
"""

import base64
import binascii
import hashlib
import json
import re
from typing import Any, List, Optional, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

from .schemas import ChatMessage

_DATA_URL_RE = re.compile(r"^data:([^;,]+)?;base64,(.+)$", re.DOTALL | re.IGNORECASE)
_IMAGE_PLACEHOLDER_RE = re.compile(r"^\[Image(?::|\])", re.IGNORECASE)
_DEFAULT_IMAGE_PROMPT = "Describe this image in detail."


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
                text = part.get("text", "")
                if text and not _IMAGE_PLACEHOLDER_RE.match(text.strip()):
                    parts.append(text)
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


def _trailing_user_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Consecutive ``user`` messages at the end (one client turn, split bubbles)."""
    trail: List[ChatMessage] = []
    for message in reversed(messages):
        if message.role == "user":
            trail.append(message)
        else:
            break
    return list(reversed(trail))


def messages_session_key(messages: List[ChatMessage]) -> Optional[str]:
    """Lookup key for an in-flight multi-turn chat, or ``None`` for a new thread."""
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


def _image_url_to_bytes(url: str) -> bytes:
    """Decode an OpenAI-style image URL (data URL or http(s)) to raw bytes."""
    url = url.strip()
    if not url:
        raise ValueError("empty image URL")
    match = _DATA_URL_RE.match(url)
    if match:
        try:
            return base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64 in data URL") from exc
    if url.startswith(("http://", "https://")):
        try:
            request = Request(url, headers={"User-Agent": "Windows-Copilot-API/1.0"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except URLError as exc:
            raise ValueError(f"could not fetch image URL: {exc}") from exc
    raise ValueError("image URL must be a data: URL or http(s) link")


def _decode_base64_payload(value: str) -> bytes:
    """Decode raw base64 or a data URL to bytes."""
    value = value.strip()
    if value.startswith("data:"):
        return _image_url_to_bytes(value)
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return base64.b64decode(value)


def _looks_like_image_mime(mime: Optional[str]) -> bool:
    return bool(mime and str(mime).lower().startswith("image/"))


def _mime_from_part(part: dict) -> Optional[str]:
    for key in ("mediaType", "mimeType", "mime_type"):
        value = part.get(key)
        if value:
            return str(value)
    return None


def _image_part_url(part: dict) -> Optional[str]:
    """Extract a URL string from one OpenAI-style multimodal content part."""
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        if isinstance(image_url, str):
            return image_url
    if part_type == "input_image":
        for key in ("image_url", "url"):
            value = part.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict) and isinstance(value.get("url"), str):
                return value["url"]
    if part_type == "image":
        image_val = part.get("image")
        if isinstance(image_val, str) and image_val.startswith(("http://", "https://", "data:")):
            return image_val
    return None


def _image_part_bytes(part: dict) -> Optional[bytes]:
    """Extract image bytes from one multimodal content part, if present."""
    url = _image_part_url(part)
    if url:
        return _image_url_to_bytes(url)

    part_type = part.get("type")
    if part_type == "image":
        # Cherry Studio / Vercel AI SDK: { type: "image", image: "<base64>", mediaType }
        image_val = part.get("image")
        if isinstance(image_val, str):
            if image_val.startswith(("http://", "https://", "data:")):
                return _image_url_to_bytes(image_val)
            return _decode_base64_payload(image_val)

    if part_type == "file":
        mime = _mime_from_part(part)
        for key in ("data", "file_data", "content"):
            value = part.get(key)
            if isinstance(value, str) and _looks_like_image_mime(mime):
                return _decode_base64_payload(value)
        file_obj = part.get("file")
        if isinstance(file_obj, dict):
            mime = mime or file_obj.get("mimeType") or file_obj.get("mime_type")
            filename = str(file_obj.get("filename") or file_obj.get("name") or "")
            if _looks_like_image_mime(mime) or filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
            ):
                for key in ("file_data", "data", "content", "url"):
                    value = file_obj.get(key)
                    if isinstance(value, str):
                        if value.startswith(("http://", "https://", "data:")):
                            return _image_url_to_bytes(value)
                        return _decode_base64_payload(value)
    return None


def _image_from_message(message: ChatMessage) -> Optional[bytes]:
    if not isinstance(message.content, list):
        return None
    for part in message.content:
        if not isinstance(part, dict):
            continue
        image_bytes = _image_part_bytes(part)
        if image_bytes:
            return image_bytes
    return None


def has_image_placeholder_without_bytes(messages: List[ChatMessage]) -> bool:
    """True when the client signalled an image but no decodable bytes arrived."""
    for message in _trailing_user_messages(messages):
        if not isinstance(message.content, list):
            continue
        has_placeholder = any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and _IMAGE_PLACEHOLDER_RE.match(part["text"].strip())
            for part in message.content
        )
        has_image_part = any(
            isinstance(part, dict)
            and part.get("type") in ("image", "image_url", "file", "input_image")
            for part in message.content
        )
        if (has_placeholder or has_image_part) and _image_from_message(message) is None:
            return True
    return False


def describe_trailing_content(messages: List[ChatMessage]) -> str:
    """Safe summary of content-part types in the current user turn (for debugging)."""
    parts = []
    for message in _trailing_user_messages(messages):
        content = message.content
        if isinstance(content, str):
            parts.append(f"str({len(content)} chars)")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "?")
                    extra = ""
                    if part_type == "text":
                        text = str(part.get("text", ""))
                        extra = f" {len(text)} chars"
                    elif part_type in ("image", "image_url", "file", "input_image"):
                        extra = " (+payload)" if _image_part_bytes(part) else " (no bytes)"
                    parts.append(f"{part_type}{extra}")
                else:
                    parts.append(type(part).__name__)
        else:
            parts.append("empty")
    return ", ".join(parts) if parts else "none"


def turn_image(messages: List[ChatMessage]) -> Optional[bytes]:
    """Return image bytes from the current client turn, if any.

    Scans consecutive trailing ``user`` messages (split image/text bubbles).
    When several images are present, only the first is forwarded.
    """
    for message in reversed(_trailing_user_messages(messages)):
        image_bytes = _image_from_message(message)
        if image_bytes:
            return image_bytes
    return None


def turn_prompt(messages: List[ChatMessage], *, image_attached: bool = False) -> str:
    """Build the Copilot prompt for one turn: optional system prefix + user text."""
    system = "\n\n".join(
        content_text(m.content) for m in messages if m.role == "system" and m.content
    )
    trailing_text = "\n\n".join(
        text
        for m in _trailing_user_messages(messages)
        if (text := content_text(m.content).strip())
    )
    users = _user_texts(messages)
    body = trailing_text or (users[-1] if users else "")
    if not body.strip() and image_attached:
        body = _DEFAULT_IMAGE_PROMPT
    if system and body:
        return f"{system}\n\n{body}"
    return system or body


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten an OpenAI ``messages`` array into a single Copilot prompt."""
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
