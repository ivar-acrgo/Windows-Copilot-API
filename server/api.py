"""FastAPI app wiring Copilot onto the OpenAI Chat Completions API."""

import sys
import threading
import time
from typing import List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from copilot import CopilotClient
from copilot.driver import ClearanceRequired

from .config import (
    DEBUG_REQUESTS,
    MODEL_ALIASES,
    MODEL_NAME,
    RATE_LIMIT_BURST,
    RATE_LIMIT_RPM,
    SESSION_TTL_SECONDS,
)
from .openai_format import (
    completion_response,
    new_id,
    sse_event,
    stream_chunk,
)
from .prompt import (
    describe_trailing_content,
    has_image_placeholder_without_bytes,
    messages_session_key,
    messages_store_key,
    turn_image,
    turn_prompt,
)
from .ratelimit import TokenBucket
from .schemas import ChatCompletionRequest, ChatMessage
from .sessions import ConversationSessionStore

app = FastAPI(title="Copilot OpenAI-compatible API", version="1.0.0")
# Server runs headless and must never pop a visible browser mid-request. With
# both recovery passes disabled, an expired clearance surfaces immediately as a
# 503 (see ClearanceRequired handling below) so an operator can re-clear out of
# band (`python -m copilot login`). Headless auto-solve is intentionally off:
# it's unreliable on low-trust egress and a failed pass can wedge the session.
client = CopilotClient(interactive_clear=False, headless_clear=False)

_CLEARANCE_HELP = (
    "Cloudflare clearance expired and could not be refreshed headlessly. "
    "Re-clear in a browser: run `python -m copilot login` (or `python tests/diagnostic.py`) "
    "and pass the 'verify you're human' check, then retry."
)

# Self-imposed rate limit on top of the concurrency lock below: this caps
# requests-per-minute, the lock caps requests-in-flight. See server/ratelimit.py.
_rate_limiter = TokenBucket(RATE_LIMIT_RPM, RATE_LIMIT_BURST)

# Maps stateless clients' message-history keys to Copilot conversation ids.
_session_store = ConversationSessionStore(SESSION_TTL_SECONDS)


def _rate_limited_response():
    """Spend a token; return an OpenAI-shaped 429 if none left, else ``None``."""
    allowed, wait = _rate_limiter.try_acquire()
    if allowed:
        return None
    secs = max(1, round(wait))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(secs)},
        content={"error": {
            "message": (
                f"Rate limit exceeded (>{RATE_LIMIT_RPM:g} req/min). "
                f"Retry in {secs}s."
            ),
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }},
    )


def _resolve_turn(
    messages: List[ChatMessage],
    explicit_conversation_id: Optional[str],
) -> Tuple[Optional[str], str, Optional[bytes]]:
    """Pick Copilot ``conversation_id``, prompt, and optional image for one turn."""
    image = turn_image(messages)
    prompt = turn_prompt(messages, image_attached=image is not None)
    if explicit_conversation_id:
        return explicit_conversation_id, prompt, image
    lookup_key = messages_session_key(messages)
    return _session_store.get(lookup_key), prompt, image


def _upstream_kwargs(image: Optional[bytes]) -> dict:
    return {"image": image} if image is not None else {}


def _vision_headers(image: Optional[bytes]) -> dict:
    return {"X-Copilot-Image-Bytes": str(len(image) if image else 0)}


def _log_request(req: ChatCompletionRequest, prompt: str, image: Optional[bytes]) -> None:
    if not DEBUG_REQUESTS:
        return
    print(
        f"[copilot-api] model={req.model!r} stream={req.stream} "
        f"image_bytes={len(image) if image else 0} "
        f"parts=[{describe_trailing_content(req.messages)}] "
        f"prompt={prompt[:80]!r}",
        file=sys.stderr,
        flush=True,
    )


def _record_turn(messages: List[ChatMessage], conversation_id: Optional[str]) -> None:
    store_key = messages_store_key(messages)
    _session_store.put(store_key, conversation_id)


# Copilot's per-account chat socket doesn't tolerate concurrent conversations
# from one process (parallel requests error out or hang). This server bridges a
# single signed-in account, so we serialize upstream calls: concurrent HTTP
# requests queue here and run one at a time. Predictable, at the cost of
# parallelism — fine for a personal bridge.
_upstream_lock = threading.Lock()


def _stream(
    messages: List[ChatMessage],
    prompt: str,
    model: str,
    conversation_id=None,
    image: Optional[bytes] = None,
):
    """Yield OpenAI ``chat.completion.chunk`` SSE events for ``prompt``."""
    cid = new_id()
    created = int(time.time())
    conv_id = conversation_id
    text_parts: List[str] = []
    upstream = _upstream_kwargs(image)
    try:
        with _upstream_lock:
            yield sse_event(stream_chunk(cid, created, model, {"role": "assistant"}))
            stream = client.stream(prompt, conversation_id=conversation_id, **upstream)
            for piece in stream:
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
                    yield sse_event(stream_chunk(cid, created, model, {"content": piece}))
            conv_id = stream.conversation_id
            yield sse_event(
                stream_chunk(
                    cid, created, model, {}, finish="stop",
                    conversation_id=conv_id,
                )
            )
    except ClearanceRequired:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_CLEARANCE_HELP}]"}, finish="error")
        )
    except Exception as exc:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="error")
        )
    else:
        _record_turn(messages, conv_id)
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models():
    ids = []
    for model_id in [MODEL_NAME, *MODEL_ALIASES]:
        if model_id not in ids:
            ids.append(model_id)
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "microsoft"}
            for model_id in ids
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        conversation_id, prompt, image = _resolve_turn(req.messages, req.conversation_id)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )
    if has_image_placeholder_without_bytes(req.messages):
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": (
                    "No image bytes reached the server. The client signalled an "
                    "image (or sent [Image: ...] placeholder text) but the payload "
                    "contained no decodable image data. In Cherry Studio, re-attach "
                    "the image and check the network request includes image_url / "
                    "file.data / image fields with base64 content."
                ),
                "type": "invalid_request_error",
            }},
        )
    if not prompt.strip() and image is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no text content in messages", "type": "invalid_request_error"}},
        )
    model = req.model or MODEL_NAME
    upstream = _upstream_kwargs(image)
    _log_request(req, prompt, image)

    limited = _rate_limited_response()
    if limited is not None:
        return limited

    if req.stream:
        return StreamingResponse(
            _stream(req.messages, prompt, model, conversation_id, image=image),
            media_type="text/event-stream",
            headers=_vision_headers(image),
        )

    try:
        with _upstream_lock:
            reply = client.chat(prompt, conversation_id=conversation_id, **upstream)
    except ClearanceRequired:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": _CLEARANCE_HELP, "type": "clearance_required"}},
            headers=_vision_headers(image),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
            headers=_vision_headers(image),
        )
    _record_turn(req.messages, reply.conversation_id)
    body = completion_response(reply.text, model, reply.conversation_id)
    return JSONResponse(content=body, headers=_vision_headers(image))


@app.get("/")
def root():
    return {"service": "Copilot OpenAI-compatible API", "endpoints": ["/v1/models", "/v1/chat/completions"]}
