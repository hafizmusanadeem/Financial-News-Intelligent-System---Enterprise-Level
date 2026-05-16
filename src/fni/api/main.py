"""
src/fni/api/main.py

FastAPI application factory.

Run locally:
    uvicorn src.fni.api.main:app --reload --host 0.0.0.0 --port 8000

Run in production (via gunicorn + uvicorn workers):
    gunicorn src.fni.api.main:app \
        --workers 2 \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --timeout 60 \
        --access-logfile -

Architecture decisions in this file:

1. LIFESPAN (not deprecated @app.on_event)
   The lifespan context manager is the FastAPI-recommended way to manage
   startup/shutdown resources. It initialises the DB pool before the first
   request arrives and cleanly drains it on shutdown — no request will ever
   see an uninitialised pool. @app.on_event("startup") is deprecated in
   FastAPI 0.93+.

2. GLOBAL EXCEPTION HANDLERS
   Without these, an unhandled Python exception returns a raw HTML 500 page.
   That's fine in a browser — catastrophic if the client is JavaScript or
   a Python script expecting JSON. All unhandled exceptions are caught and
   returned as { "detail": "..." } JSON with a proper HTTP status code.

3. REQUEST LOGGING MIDDLEWARE
   Every request logs: method, path, status code, duration.
   In production this feeds into your log aggregator (Datadog, CloudWatch,
   Loki). Without it, you have no visibility into which endpoints are slow
   or which are being called most.

4. CORS
   Configured with allow_origins from the environment. In development,
   set CORS_ORIGINS=* in .env. In production, set the exact frontend domain.
   Wildcard CORS in production is a security vulnerability.

5. API VERSIONING (/api/v1/)
   All data routes are prefixed /api/v1. When you break the schema in the
   future (rename a field, remove an endpoint), /api/v2 can coexist while
   analysts migrate their tooling. Skipping versioning now means a breaking
   change = emergency for every consumer simultaneously.

6. DOCS DISABLED IN PRODUCTION
   Swagger UI (/docs) and ReDoc (/redoc) expose your full schema publicly.
   In production, disable them or put them behind auth. The env flag
   ENABLE_DOCS controls this.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.fni.api.dependencies import close_pool, init_pool
from src.fni.api.routers import analytics, health, news
from src.fni.core.config import settings
from src.fni.core.logger import setup_logger, get_logger

load_dotenv()
setup_logger()
logger = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.
    """
    logger.info("API starting up ...")
    init_pool()           # warm DB connections before first request
    logger.info("API ready.")
    yield
    logger.info("API shutting down ...")
    close_pool()          # drain connections gracefully
    logger.info("API shutdown complete.")


# ── App factory ────────────────────────────────────────────────────────────────

_enable_docs = os.getenv("ENABLE_DOCS", "true").lower() == "true"

app = FastAPI(
    title="Financial News Intelligence API",
    description=(
        "Serves processed financial news, sentiment scores, LLM event labels, "
        "and next-day price impact data for analyst decision support."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
)


# ── CORS ───────────────────────────────────────────────────────────────────────

_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET"],          # API is read-only — no POST/PUT/DELETE needed
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next: object) -> object:
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    # Attach request ID so logs can be correlated per-request
    request.state.request_id = request_id

    from starlette.middleware.base import _CachedRequest  # type: ignore[import]
    response = await call_next(request)  # type: ignore[operator]

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.1f}ms)"
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handlers ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches any unhandled exception and returns JSON 500.
    Without this, FastAPI returns an HTML error page — unusable for API clients.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"[{request_id}] Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal error occurred. Check server logs.",
            "request_id": request_id,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: object) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Route '{request.url.path}' not found."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(news.router)
app.include_router(analytics.router)


# ── Root redirect ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse({
        "service": "Financial News Intelligence API",
        "version": settings.VERSION,
        "docs": "/docs" if _enable_docs else "disabled",
        "health": "/health",
    })