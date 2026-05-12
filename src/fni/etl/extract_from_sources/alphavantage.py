"""
src/fni/etl/extract_from_sources/alphavantage.py

Pulls raw news from the AlphaVantage NEWS_SENTIMENT endpoint,
deduplicates, and saves to a CSV for the next pipeline stage.
"""

import os
import sys
import ast
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException
from src.fni.core.constants import TICKERS, SLEEP_BETWEEN, LIMIT, TIME_FROM, TIME_TO

load_dotenv()
setup_logger()
logger = get_logger(__name__)


class AlphaVantageExtractor:

    def __init__(
        self,
        tickers: list[str],
        time_from: str,
        time_to: str,
        save_path: str = "artifacts/interim/news_data.csv",
        limit: int = LIMIT,
        sleep_between: int = SLEEP_BETWEEN,
    ) -> None:
        self.api_key: str | None = os.getenv("API_alphavantage")
        self.tickers: list[str] = tickers
        self.time_from: str = time_from
        self.time_to: str = time_to
        self.save_path: Path = Path(save_path)
        self.limit: int = limit
        self.sleep_between: int = sleep_between

        if not self.api_key:
            raise CustomException(
                "API_alphavantage is not set in .env. Add it before running.", sys
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _process_raw_df(self, df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Minimal normalisation applied at extraction time:
        - Attach ticker column
        - Parse time_published to datetime
        - Keep only the columns downstream stages need
        """
        try:
            df = df_raw.copy()
            df["ticker"] = ticker
            df["time_published"] = pd.to_datetime(
                df["time_published"],
                format="%Y%m%dT%H%M%S",
                errors="coerce",
            )
            KEEP: list[str] = [
                "ticker", "title", "summary", "source",
                "time_published", "overall_sentiment_score",
                "overall_sentiment_label",
            ]
            return df[[c for c in KEEP if c in df.columns]]
        except Exception as e:
            raise CustomException(
                f"Failed to process dataframe for {ticker}: {e}", sys
            ) from e

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────

    def _fetch(self) -> pd.DataFrame:
        logger.info(
            f"Starting batch | {self.time_from} → {self.time_to} "
            f"| {len(self.tickers)} tickers"
        )
        frames: list[pd.DataFrame] = []

        for i, ticker in enumerate(self.tickers, 1):
            try:
                url = (
                    f"https://www.alphavantage.co/query"
                    f"?function=NEWS_SENTIMENT"
                    f"&tickers={ticker}"
                    f"&time_from={self.time_from}"
                    f"&time_to={self.time_to}"
                    f"&limit={self.limit}"
                    f"&apikey={self.api_key}"
                )
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data: dict[str, object] = response.json()  # type: ignore[assignment]

                if "feed" not in data:
                    logger.warning(
                        f"[{i}/{len(self.tickers)}] {ticker} — no feed returned: {data}"
                    )
                    time.sleep(self.sleep_between)
                    continue

                df_raw = pd.DataFrame(data["feed"])
                df_processed = self._process_raw_df(df_raw, ticker)
                logger.info(
                    f"[{i}/{len(self.tickers)}] {ticker} "
                    f"— raw: {len(df_raw)} → kept: {len(df_processed)}"
                )
                frames.append(df_processed)

            except CustomException:
                raise
            except Exception as e:
                raise CustomException(
                    f"Request failed for {ticker}: {e}", sys
                ) from e

            time.sleep(self.sleep_between)

        if not frames:
            raise CustomException(
                "Nothing fetched across all tickers. "
                "Check API key or date range.", sys
            )

        df_new = pd.concat(frames, ignore_index=True)
        logger.info(
            f"Batch complete — shape: {df_new.shape} | "
            f"range: {df_new['time_published'].min()} → {df_new['time_published'].max()}"
        )
        return df_new

    # ── Step 2: Append ────────────────────────────────────────────────────────

    def _append_and_save(
        self, df_new: pd.DataFrame
    ) -> tuple[int, pd.DataFrame, int]:
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.save_path.exists():
                df_existing = pd.read_csv(self.save_path)
                rows_before = len(df_existing)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                logger.info("No existing file found — creating fresh.")
                rows_before = 0
                df_combined = df_new

            before_dedup = len(df_combined)
            df_combined = df_combined.drop_duplicates(
                subset=["ticker", "title", "time_published"]
            ).reset_index(drop=True)
            dupes_removed = before_dedup - len(df_combined)

            df_combined.to_csv(self.save_path, index=False)
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Failed during append/save: {e}", sys) from e

        return rows_before, df_new, dupes_removed

    # ── Step 3: Verify ────────────────────────────────────────────────────────

    def _verify(
        self,
        rows_before: int,
        df_new: pd.DataFrame,
        dupes_removed: int,
    ) -> pd.DataFrame:
        try:
            df_verify = pd.read_csv(self.save_path)
            logger.info(
                f"Verification — before: {rows_before} | new: {len(df_new)} | "
                f"dupes removed: {dupes_removed} | final: {len(df_verify)}"
            )
            logger.info(
                f"Date range in file: "
                f"{df_verify['time_published'].min()} → {df_verify['time_published'].max()}"
            )
            logger.info(
                f"Rows per ticker:\n{df_verify['ticker'].value_counts().to_string()}"
            )
            null_counts = df_verify.isnull().sum()
            if null_counts.any():
                logger.warning(
                    f"Nulls detected:\n{null_counts[null_counts > 0].to_string()}"
                )
            return df_verify
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(f"Verification failed: {e}", sys) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        df_new = self._fetch()
        rows_before, df_new, dupes_removed = self._append_and_save(df_new)
        return self._verify(rows_before, df_new, dupes_removed)


if __name__ == "__main__":
    extractor = AlphaVantageExtractor(
        tickers=TICKERS,
        time_from=TIME_FROM,
        time_to=TIME_TO,
    )
    extractor.run()
