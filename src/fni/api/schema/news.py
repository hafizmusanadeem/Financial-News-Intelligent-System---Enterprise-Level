"""
src/fni/api/schemas/news.py

Pydantic v2 response models for all /news endpoints.

Design decisions:
  - model_config = ConfigDict(from_attributes=True) lets us pass psycopg2
    RealDictRow objects directly into model_validate() without manual dict conversion.
  - All financial floats are float | None — DB rows without price data (news
    published on weekends, or before impact-score stage ran) must not crash the API.
  - No input/request schemas here — query params are validated inline in the
    router using FastAPI's Query() with constraints. Pydantic models for query
    params would add boilerplate with no gain at this scale.
"""

from __future__ import annotations

from datetime import date, datetime
# from typing import Annotated # Annotated is Yet to Add

from pydantic import BaseModel, ConfigDict, Field


class NewsItem(BaseModel):
    """
    Full representation of a single news row.
    Returned by GET /news and GET /news/{ticker}.
    """

    model_config = ConfigDict(from_attributes=True)

    ticker: str
    time_published: datetime
    news_date: date | None = None
    price_date: date | None = None
    title: str
    summary: str
    source: str | None = None

    # Sentiment (from AlphaVantage)
    overall_sentiment_score: float | None = None
    overall_sentiment_label: str | None = None

    # LLM classification
    event_label: str | None = None

    # Impact (computed against Yahoo Finance prices)
    day0_close: float | None = None
    day1_close: float | None = None
    next_day_return: float | None = None
    impact_score: float | None = None
    impact_label: str | None = None

    # OHLC snapshot (day1)
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None

    inserted_at: datetime | None = None


class PaginatedNewsResponse(BaseModel):
    """Envelope returned by the paginated news list endpoint."""

    total: int = Field(..., description="Total matching rows (ignoring pagination)")
    page: int
    page_size: int
    results: list[NewsItem]