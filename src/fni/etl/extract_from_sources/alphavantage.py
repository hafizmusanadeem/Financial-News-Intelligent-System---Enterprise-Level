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

setup_logger()
get_logger('INFO')
logger = get_logger(__name__)

load_dotenv()
# ============================================================
# CONFIG — only change this block between runs
# ============================================================
API_KEY       = os.getenv('API_alphavantage')
TICKERS = [
    "AAPL", "AMZN", "MSFT", "GOOGL", "NVDA",
    "META", "TSLA", "JPM",  "V",     "JNJ",
    "BAC",  "WMT",  "UNH",  "XOM",   "CVX",
    "HD",   "PG",   "MA",   "ABBV",  "MRK"
]
TIME_FROM     = "20260101T0000"
TIME_TO       = "20260105T0000"
LIMIT         = 50
SAVE_PATH     = Path("data/alphavantage_news.csv")
SLEEP_BETWEEN = 13

if not API_KEY:
    raise CustomException("API_KEY is not set. Update the API_KEY before running.", sys)
# ============================================================


def parse_if_string(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return []
    return val if val is not None else []


def process_raw_df(df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
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


# ── Step 1: Fetch ─────────────────────────────────────────────────────────

logger.info(f"Starting batch | {TIME_FROM} → {TIME_TO} | {len(TICKERS)} tickers")

frames = []

for i, ticker in enumerate(TICKERS, 1):
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT"
            f"&tickers={ticker}"
            f"&time_from={TIME_FROM}"
            f"&time_to={TIME_TO}"
            f"&limit={LIMIT}"
            f"&apikey={API_KEY}"
        )

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "feed" not in data:
            logger.warning(f"[{i}/{len(TICKERS)}] {ticker} — no feed returned: {data}")
            time.sleep(SLEEP_BETWEEN)
            continue

        df_raw = pd.DataFrame(data["feed"])
        df_processed = process_raw_df(df_raw, ticker)

        logger.info(f"[{i}/{len(TICKERS)}] {ticker} — raw: {len(df_raw)} → kept: {len(df_processed)}")
        frames.append(df_processed)

    except Exception as e:
        raise CustomException(f"Request failed for {ticker}: {e}", sys)

    time.sleep(SLEEP_BETWEEN)

if not frames:
    raise CustomException("Nothing fetched across all tickers. Check API key or date range.", sys)

df_new = pd.concat(frames, ignore_index=True)
logger.info(f"Batch complete — shape: {df_new.shape} | range: {df_new['time_published'].min()} → {df_new['time_published'].max()}")


# ── Step 2: Append ────────────────────────────────────────────────────────

try:
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_PATH.exists():
        df_existing = pd.read_csv(SAVE_PATH)
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

    df_combined.to_csv(SAVE_PATH, index=False)

except Exception as e:
    raise CustomException(f"Failed during append/save: {e}", sys)


# ── Step 3: Verification ──────────────────────────────────────────────────

try:
    df_verify = pd.read_csv(SAVE_PATH)

    logger.info(
        f"Verification — before: {rows_before} | new: {len(df_new)} | "
        f"dupes removed: {dupes_removed} | final: {len(df_verify)}"
    )
    logger.info(f"Date range in file: {df_verify['time_published'].min()} → {df_verify['time_published'].max()}")
    logger.info(f"Rows per ticker:\n{df_verify['ticker'].value_counts().to_string()}")

    null_counts = df_verify.isnull().sum()
    if null_counts.any():
        logger.warning(f"Nulls detected:\n{null_counts[null_counts > 0].to_string()}")

except Exception as e:
    raise CustomException(f"Verification failed: {e}", sys)