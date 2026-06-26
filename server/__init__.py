"""OpenAI-compatible HTTP server for Microsoft Copilot.

Start it:

    from server import app
    app()

(`python app.py` in the project root does exactly this.) The server listens on
``0.0.0.0:8000`` by default (all IPv4 interfaces). Set ``HOST`` / ``PORT`` to
override — e.g. ``HOST=127.0.0.1`` for local-only, ``HOST=::`` for IPv6.
Completions shape onto :class:`copilot.CopilotClient`; sign in once first with
``python -m copilot login``.

Code is split by concern:

    config.py         constants
    schemas.py        pydantic request models
    prompt.py         OpenAI messages -> per-turn Copilot prompt + session keys
    sessions.py       in-memory conversation_id cache for stateless clients
    openai_format.py  build OpenAI response/chunk shapes
    api.py            FastAPI app, routes, upstream serialization
"""

import os

from .api import app as _api
from .config import HOST as DEFAULT_HOST, PORT as DEFAULT_PORT


def _listen_url(host: str, port: int) -> str:
    if host == "0.0.0.0":
        return f"http://{host}:{port}  (all IPv4 interfaces)"
    if host == "::":
        return f"http://[{host}]:{port}  (all IPv6 interfaces)"
    return f"http://{host}:{port}"


def app(host=None, port=None) -> None:
    """Start the server (blocks while uvicorn runs).

    On first run (no saved session) this opens a browser for interactive sign-in
    before serving, so requests don't fail with a "not signed in" error.
    """
    import uvicorn

    from copilot.auth import load_auth

    if host is None:
        host = os.environ.get("HOST", DEFAULT_HOST)
    if port is None:
        port = int(os.environ.get("PORT", str(DEFAULT_PORT)))

    # Use cached session/token.json when fresh; never pop a browser from the
    # server process (headless Pi has no Playwright — copy token.json from a PC).
    try:
        load_auth(auto_login=False)
    except Exception as exc:
        print(f"Warning: could not establish a Copilot session: {exc}")

    print(
        f"Copilot OpenAI-compatible API on {_listen_url(host, port)}  "
        f"(POST /v1/chat/completions)"
    )
    uvicorn.run(_api, host=host, port=port)


__all__ = ["app"]
