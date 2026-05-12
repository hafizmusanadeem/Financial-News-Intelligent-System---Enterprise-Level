"""
src/fni/etl/transform/clean.py

Post-extraction cleaning utilities.

NOTE ON DESIGN:
    Basic normalisation (column selection, datetime parsing) is intentionally
    done inside AlphaVantageExtractor._process_raw_df() so the extractor always
    produces a consistent schema regardless of caller.

    This module handles *additional* cleaning that runs after extraction and
    before LLM labelling:
        - Drop rows with null title/summary (LLM input would be meaningless)
        - Strip whitespace from text columns
        - Deduplicate on (ticker, title) if extractor dedup was bypassed
        - Enforce correct dtypes before the labeller reads the CSV

    It is called by pipeline.py as an explicit stage between extract and label.
"""

import ast
import sys
from typing import Any

import pandas as pd

from src.fni.core.logger import get_logger
from src.fni.core.exceptions import CustomException

logger = get_logger(__name__)


# ── Helpers (used by both this module and impact_score) ──────────────────────

def parse_if_string(val: Any) -> list[Any]:
    """Safely parse a stringified list/dict back to a Python object."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)  # type: ignore[return-value]
        except (ValueError, SyntaxError):
            return []
    return val if val is not None else []


def get_target_relevance(ticker_sentiment: Any, target_ticker: str) -> float:
    for item in parse_if_string(ticker_sentiment):
        if isinstance(item, dict) and item.get("ticker") == target_ticker:
            return float(item.get("relevance_score", 0.0))
    return 0.0


# ── Cleaner ───────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Cleans the raw interim CSV produced by AlphaVantageExtractor.

    Responsibilities (only things NOT already done by the extractor):
        1. Drop rows where title or summary is null / empty string
        2. Strip leading/trailing whitespace from text columns
        3. Remove exact-duplicate (ticker, title) pairs that slipped through
        4. Re-cast overall_sentiment_score to float (read_csv may infer object)
        5. Log a cleaning summary for audit purposes
    """

    TEXT_COLS: list[str] = ["title", "summary", "source"]
    REQUIRED_COLS: list[str] = ["ticker", "title", "summary", "time_published"]

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
            logger.info(f"[CLEAN] Loaded {len(df)} rows from {self.input_path}")
            return df
        except Exception as e:
            raise CustomException(f"DataCleaner failed to load input: {e}", sys) from e

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise CustomException(
                f"[CLEAN] Input CSV is missing required columns: {missing}", sys
            )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_in = len(df)

        # 1. Strip whitespace on text columns
        for col in self.TEXT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # 2. Drop rows with empty/null title or summary
        #    (LLM labeller would receive empty prompts — no value in keeping them)
        df = df[
            df["title"].notna() & (df["title"] != "") & (df["title"] != "nan") &
            df["summary"].notna() & (df["summary"] != "") & (df["summary"] != "nan")
        ].reset_index(drop=True)
        dropped_null = rows_in - len(df)

        # 3. Deduplicate on (ticker, title) — covers any edge case from extractor
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["ticker", "title"]).reset_index(drop=True)
        dropped_dedup = before_dedup - len(df)

        # 4. Re-cast sentiment score
        if "overall_sentiment_score" in df.columns:
            df["overall_sentiment_score"] = pd.to_numeric(
                df["overall_sentiment_score"], errors="coerce"
            )

        # 5. Ensure time_published is parsed correctly
        df["time_published"] = pd.to_datetime(
            df["time_published"], errors="coerce"
        )
        dropped_ts = df["time_published"].isna().sum()
        df = df.dropna(subset=["time_published"]).reset_index(drop=True)

        logger.info(
            f"[CLEAN] rows_in={rows_in} | dropped_null={dropped_null} | "
            f"dropped_dedup={dropped_dedup} | dropped_bad_ts={dropped_ts} | "
            f"rows_out={len(df)}"
        )
        return df

    def _save(self, df: pd.DataFrame) -> None:
        try:
            import pathlib
            pathlib.Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_path, index=False)
            logger.info(f"[CLEAN] Saved {len(df)} clean rows → {self.output_path}")
        except Exception as e:
            raise CustomException(f"DataCleaner failed to save output: {e}", sys) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        df = self._load()
        self._validate_schema(df)
        df_clean = self._clean(df)
        self._save(df_clean)
        return df_clean