"""Server configuration — shared constants."""

import os

# The model id(s) this bridge advertises. ``gpt-4o`` is listed so clients like
# Cherry Studio auto-detect vision support (their isVisionModel regex matches
# gpt-4*). Any model name still works on /v1/chat/completions.
MODEL_NAME = os.environ.get("MODEL_NAME", "copilot")
MODEL_ALIASES = [
    alias.strip()
    for alias in os.environ.get("MODEL_ALIASES", "gpt-4o").split(",")
    if alias.strip()
]

# Self-imposed rate limit (Copilot publishes none). Tune to whatever ceiling the
# probe in tests/ratelimit.py shows your account tolerates.
#   RATE_LIMIT_RPM   requests/minute the bridge will accept; 0 disables limiting.
#   RATE_LIMIT_BURST max requests allowed back-to-back before pacing kicks in.
# Default 12 rpm sits safely below the ~15 rpm where one account starts seeing
# upstream 502s, so the limiter only bites when callers try to exceed that.
RATE_LIMIT_RPM = float(os.environ.get("RATE_LIMIT_RPM", "12"))  # 12 rpm ≈ 5s per call
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "4"))

# How long to remember Copilot conversation ids for stateless OpenAI clients that
# resend full message history without conversation_id. 0 disables auto-mapping.
SESSION_TTL_SECONDS = float(os.environ.get("SESSION_TTL_SECONDS", str(24 * 3600)))

# Log a one-line summary of each /v1/chat/completions request to stderr.
DEBUG_REQUESTS = os.environ.get("DEBUG_REQUESTS", "").lower() in ("1", "true", "yes")
