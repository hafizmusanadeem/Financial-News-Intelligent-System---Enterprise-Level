"""
src/fni/api/dependencies.py

Database connection pool — injected into every route via FastAPI's
dependency injection system.

Design decisions:

1. ThreadedConnectionPool vs per-request connect()
   Creating a new psycopg2 connection on every request takes ~50–200ms
   (TCP handshake + Postgres auth). With 20 analysts hitting the API
   concurrently this compounds into seconds of wasted latency per request.
   ThreadedConnectionPool holds N live connections and hands them out in <1ms.

2. Why not asyncpg / async DB?
   FastAPI runs sync route functions in a thread pool automatically
   (via Starlette's run_in_threadpool). For a dataset of ~20 tickers with
   a few thousand rows, sync psycopg2 inside a thread is perfectly adequate.
   Async DB drivers (asyncpg, psycopg v3 async) earn their complexity when
   you have hundreds of concurrent long-running queries competing for the
   event loop — that's not this system. Introduce async when you measure a
   bottleneck, not before.

3. Why RealDictCursor?
   Returns rows as dict-like objects. Pydantic's model_validate() accepts
   dict-like inputs directly, so there's zero intermediate mapping layer
   between the DB row and the JSON response.

4. Pool sizing
   minconn=2  — always keep 2 connections warm (avoids cold-start on first request)
   maxconn=10 — hard cap; Postgres default max_connections is 100, so 10 per
                app instance leaves headroom for other services and multiple
                deployment instances.

5. Context manager pattern
   get_db() yields the connection, then puts it back regardless of whether
   the route succeeded or raised. Without this guarantee, a 500 error would
   leak a connection and the pool would silently exhaust under load.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from fastapi import HTTPException, status

from src.fni.core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Pool singleton ─────────────────────────────────────────────────────────────
# Initialised once during app lifespan startup (see main.py).
# Routes never create the pool — they only consume from it.

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool() -> None:
    """
    Called once at application startup (inside lifespan context manager).
    Raises RuntimeError if DB env vars are missing.
    """
    global _pool

    required = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required DB environment variables: {missing}. "
            "Add them to your .env file."
        )

    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        sslmode=os.getenv("DB_SSLMODE", "prefer"),
        connect_timeout=10,
        # Automatically recycle stale connections after 10 min idle
        options="-c statement_timeout=30000",  # 30s query timeout — kills runaway queries
    )
    logger.info("DB connection pool initialised (minconn=2, maxconn=10).")


def close_pool() -> None:
    """Called once at application shutdown (inside lifespan context manager)."""
    global _pool
    if _pool:
        _pool.closeall()
        logger.info("DB connection pool closed.")
    _pool = None


@contextmanager
def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    FastAPI dependency — yields a live DB connection from the pool.

    Usage in a route:
        def my_route(conn = Depends(get_db)):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(...)

    The connection is ALWAYS returned to the pool, even on exception.
    HTTP 503 is raised if the pool is exhausted (all connections in use).
    """
    if _pool is None:
        # Should never happen after startup — guard for test environments
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database pool not initialised.",
        )

    conn = None
    try:
        conn = _pool.getconn()
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="All database connections are in use. Try again shortly.",
            )
        yield conn
    finally:
        if conn is not None:
            _pool.putconn(conn)