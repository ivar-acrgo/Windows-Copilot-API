"""Start the OpenAI-compatible Copilot server:

    python app.py            # listens on 0.0.0.0:8000 (set HOST / PORT to override)

To bind local-only or IPv6:

    HOST=127.0.0.1 python app.py
    HOST=:: python app.py

Or point uvicorn at the ASGI app directly:

    uvicorn server.api:app --host 0.0.0.0 --port 8080
"""

from server import app

if __name__ == "__main__":
    app()  # blocks while uvicorn runs
