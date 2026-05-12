# =============================================================================
# src/fni/core/constants.py
# Single source of truth for all project-wide constants.
# DB credentials live in .env
# =============================================================================

# ── AlphaVantage ──────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD: float = 0.6
SLEEP_BETWEEN: int = 13
LIMIT: int = 50
TIME_FROM: str = "20260101T0000"
TIME_TO: str = "20260110T0000"

TICKERS: list[str] = [
    "AAPL", "AMZN", "MSFT", "GOOGL", "NVDA",
    "META", "TSLA", "JPM",  "V",     "JNJ",
    "BAC",  "WMT",  "UNH",  "XOM",   "CVX",
    "HD",   "PG",   "MA",   "ABBV",  "MRK",
]

# ── LLM Labeller ─────────────────────────────────────────────────────────────

REQUEST_DELAY: int = 3   # seconds between OpenRouter calls

# ── Database ──────────────────────────────────────────────────────────────────
# All DB credentials are loaded from .env via etl/load/configuration.py.
# Required keys: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT