import requests
import pandas as pd
import time
import ast
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException
from src.fni.core.constants import TICKERS_LABELS

setup_logger()
get_logger('INFO')
logger = get_logger(__name__)

load_dotenv()

TICKERS = TICKERS_LABELS
class AlphaVantageExtractor:

    def __init__(
        self,
        tickers: list,
        time_from: str,
        time_to: str,
        save_path: str = "data/alphavantage_news.csv",
        limit: int = 50,
        sleep_between: int = 13
    ):
        self.api_key       = os.getenv('API_alphavantage')
        self.tickers       = tickers
        self.time_from     = time_from
        self.time_to       = time_to
        self.save_path     = Path(save_path)
        self.limit         = limit
        self.sleep_between = sleep_between

        if not self.api_key:
            raise CustomException("API_KEY is not set. Update the API_KEY before running.", sys)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _parse_if_string(self, val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return []
        return val if val is not None else []

    def _process_raw_df(self, df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        try:
            df = df_raw.copy()
            df["ticker"] = ticker

            df["time_published"] = pd.to_datetime(
                df["time_published"],
                format="%Y%m%dT%H%M%S"
            )

            KEEP = [
                "ticker", "title", "summary", "source",
                "time_published", "overall_sentiment_score",
                "overall_sentiment_label"
            ]
            return df[[c for c in KEEP if c in df.columns]]

        except Exception as e:
            raise CustomException(f"Failed to process dataframe for {ticker}: {e}", sys)

    # ── Step 1: Fetch ─────────────────────────────────────────────────────

    def _fetch(self) -> pd.DataFrame:
        logger.info(f"Starting batch | {self.time_from} → {self.time_to} | {len(self.tickers)} tickers")

        frames = []

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

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if "feed" not in data:
                    logger.warning(f"[{i}/{len(self.tickers)}] {ticker} — no feed returned: {data}")
                    time.sleep(self.sleep_between)
                    continue

                df_raw = pd.DataFrame(data["feed"])
                df_processed = self._process_raw_df(df_raw, ticker)

                logger.info(f"[{i}/{len(self.tickers)}] {ticker} — raw: {len(df_raw)} → kept: {len(df_processed)}")
                frames.append(df_processed)

            except Exception as e:
                raise CustomException(f"Request failed for {ticker}: {e}", sys)

            time.sleep(self.sleep_between)

        if not frames:
            raise CustomException("Nothing fetched across all tickers. Check API key or date range.", sys)

        df_new = pd.concat(frames, ignore_index=True)
        logger.info(f"Batch complete — shape: {df_new.shape} | range: {df_new['time_published'].min()} → {df_new['time_published'].max()}")

        return df_new

    # ── Step 2: Append ────────────────────────────────────────────────────

    def _append_and_save(self, df_new: pd.DataFrame):
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.save_path.exists():
                df_existing = pd.read_csv(self.save_path)
                rows_before = len(df_existing)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                logger.info("No existing file found — creating fresh.")
                df_existing = pd.DataFrame()
                rows_before = 0
                df_combined = df_new

            before_dedup  = len(df_combined)
            df_combined   = df_combined.drop_duplicates(subset=["ticker", "title", "time_published"]).reset_index(drop=True)
            dupes_removed = before_dedup - len(df_combined)

            df_combined.to_csv(self.save_path, index=False)

        except Exception as e:
            raise CustomException(f"Failed during append/save: {e}", sys)

        return rows_before, df_new, dupes_removed

    # ── Step 3: Verification ──────────────────────────────────────────────

    def _verify(self, rows_before: int, df_new: pd.DataFrame, dupes_removed: int):
        try:
            df_verify = pd.read_csv(self.save_path)

            logger.info(
                f"Verification — before: {rows_before} | new: {len(df_new)} | "
                f"dupes removed: {dupes_removed} | final: {len(df_verify)}"
            )
            logger.info(f"Date range in file: {df_verify['time_published'].min()} → {df_verify['time_published'].max()}")
            logger.info(f"Rows per ticker:\n{df_verify['ticker'].value_counts().to_string()}")

            null_counts = df_verify.isnull().sum()
            if null_counts.any():
                logger.warning(f"Nulls detected:\n{null_counts[null_counts > 0].to_string()}")

            return df_verify

        except Exception as e:
            raise CustomException(f"Verification failed: {e}", sys)

    # ── Public entry point ────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        df_new                            = self._fetch()
        rows_before, df_new, dupes_removed = self._append_and_save(df_new)
        return self._verify(rows_before, df_new, dupes_removed)


if __name__ == "__main__":
    extractor = AlphaVantageExtractor(
        tickers=TICKERS,
        time_from="20240101T0000",
        time_to="20250101T0000"
    )
    extractor.run()