"""
src/fni/etl/load/load.py

Loads the final transformed + impact-scored DataFrame into PostgreSQL.

Table  : financial_news
Key    : (ticker, time_published)
Strategy: COPY into a per-transaction temp table → upsert into main table.
          Fully idempotent — safe to re-run on the same data.
"""

import io
import sys
from typing import Final

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.fni.core.logger import get_logger
from src.fni.core.exceptions import CustomException
from src.fni.etl.load.configuration import get_db_connection

logger = get_logger(__name__)

TABLE_NAME: Final[str] = "financial_news"

CREATE_TABLE_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                      SERIAL,
    ticker                  VARCHAR(20)     NOT NULL,
    time_published          TIMESTAMPTZ     NOT NULL,
    news_date               DATE,
    price_date              DATE,
    title                   TEXT,
    summary                 TEXT,
    source                  VARCHAR(255),
    overall_sentiment_score NUMERIC(6, 4),
    overall_sentiment_label VARCHAR(50),
    event_label             VARCHAR(80),
    day0_close              NUMERIC(12, 4),
    day1_close              NUMERIC(12, 4),
    next_day_return         NUMERIC(10, 6),
    impact_score            NUMERIC(6, 4),
    impact_label            VARCHAR(20),
    open_price              NUMERIC(12, 4),
    high_price              NUMERIC(12, 4),
    low_price               NUMERIC(12, 4),
    inserted_at             TIMESTAMPTZ     DEFAULT NOW(),
    PRIMARY KEY (ticker, time_published)
);
"""

# DataFrame column → DB column name
COLUMN_MAP: Final[dict[str, str]] = {
    "ticker":                   "ticker",
    "time_published":           "time_published",
    "news_date":                "news_date",
    "price_date":               "price_date",
    "title":                    "title",
    "summary":                  "summary",
    "source":                   "source",
    "overall_sentiment_score":  "overall_sentiment_score",
    "overall_sentiment_label":  "overall_sentiment_label",
    "event_label":              "event_label",
    "day0_close":               "day0_close",
    "day1_close":               "day1_close",
    "next_day_return":          "next_day_return",
    "impact_score":             "impact_score",
    "impact_label":             "impact_label",
    "Open":                     "open_price",
    "High":                     "high_price",
    "Low":                      "low_price",
}


class NewsLoader:
    """
    Bulk-loads a transformed financial news DataFrame into PostgreSQL.

    Flow per call to load():
        _ensure_schema()          → CREATE TABLE IF NOT EXISTS
        _prepare_df()             → rename/select columns, fix dtypes
        _bulk_copy_to_temp()      → COPY into temp table (fast, lock-free on main)
        _upsert_from_temp()       → INSERT … ON CONFLICT DO UPDATE
        conn.commit()
    """

    def __init__(self, conn: PgConnection | None = None) -> None:
        self._conn: PgConnection = conn if conn is not None else get_db_connection()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        try:
            with self._conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            self._conn.commit()
            logger.info(f"Table '{TABLE_NAME}' verified / created.")
        except Exception as e:
            self._conn.rollback()
            raise CustomException(f"Schema creation failed: {e}", sys) from e

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        available: dict[str, str] = {
            k: v for k, v in COLUMN_MAP.items() if k in df.columns
        }
        if not available:
            raise CustomException(
                "No recognisable columns in DataFrame — cannot load.", sys
            )

        out: pd.DataFrame = df[list(available.keys())].rename(columns=available).copy()

        if "time_published" in out.columns:
            out["time_published"] = pd.to_datetime(
                out["time_published"], utc=True, errors="coerce"
            )
        if "news_date" in out.columns:
            out["news_date"] = pd.to_datetime(
                out["news_date"], errors="coerce"
            ).dt.date
        if "price_date" in out.columns:
            out["price_date"] = pd.to_datetime(
                out["price_date"], errors="coerce"
            ).dt.date

        # Drop rows without the primary key values — they would fail the constraint
        before = len(out)
        out = out.dropna(subset=["ticker", "time_published"]).reset_index(drop=True)
        dropped = before - len(out)
        if dropped:
            logger.warning(
                f"[LOAD] Dropped {dropped} rows with null PK values "
                f"(ticker or time_published)."
            )

        return out

    def _bulk_copy_to_temp(self, df: pd.DataFrame) -> str:
        temp_table: str = f"_tmp_{TABLE_NAME}"
        col_list: str = ", ".join(df.columns)

        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)

        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TEMP TABLE IF NOT EXISTS {temp_table}
                (LIKE {TABLE_NAME} INCLUDING DEFAULTS)
                ON COMMIT DROP;
            """)
            cur.copy_expert(
                f"COPY {temp_table} ({col_list}) FROM STDIN WITH CSV NULL '\\N'",
                buf,
            )

        logger.info(f"[LOAD] COPY loaded {len(df)} rows into temp table.")
        return temp_table

    def _upsert_from_temp(self, temp_table: str, db_cols: list[str]) -> int:
        pk_cols: set[str] = {"ticker", "time_published"}
        update_cols: list[str] = [c for c in db_cols if c not in pk_cols]
        set_clause: str = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        col_list: str = ", ".join(db_cols)

        upsert_sql: str = f"""
            INSERT INTO {TABLE_NAME} ({col_list})
            SELECT {col_list} FROM {temp_table}
            ON CONFLICT (ticker, time_published)
            DO UPDATE SET {set_clause};
        """
        with self._conn.cursor() as cur:
            cur.execute(upsert_sql)
            return cur.rowcount

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("[LOAD] Empty DataFrame — nothing to load.")
            return

        try:
            self._ensure_schema()
            prepared: pd.DataFrame = self._prepare_df(df)

            if prepared.empty:
                logger.warning("[LOAD] All rows dropped during preparation — nothing to load.")
                return

            temp_table: str = self._bulk_copy_to_temp(prepared)
            affected: int = self._upsert_from_temp(temp_table, list(prepared.columns))
            self._conn.commit()

            logger.info(
                f"[LOAD] Complete — {affected} rows upserted into '{TABLE_NAME}' "
                f"(input rows: {len(df)})."
            )

        except CustomException:
            self._conn.rollback()
            raise
        except Exception as e:
            self._conn.rollback()
            raise CustomException(f"Load failed: {e}", sys) from e

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()
            logger.info("[LOAD] PostgreSQL connection closed.")