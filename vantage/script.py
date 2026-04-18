import requests
import pandas as pd
import time
import os
import ast
from pathlib import Path

# ============================================================
# CONFIG — only edit this block between runs
# ============================================================
API_KEY = '9K5BTZTGHH2M1P76'
TICKERS   = [
    "AAPL", "AMZN", "MSFT", "GOOGL", "NVDA",
    "META", "TSLA", "JPM", "V",    "JNJ",
    "BAC",  "WMT",  "UNH",  "XOM",  "CVX",
    "HD",   "PG",   "MA",   "ABBV", "MRK"
]
TIME_FROM           = "20250101T0000"   # change each batch
TIME_TO             = "20250401T0000"   # change each batch
LIMIT               = 50
SAVE_PATH           = Path("vantage/alphavantage_news.csv")
RELEVANCE_THRESHOLD = 0.73
SLEEP_BETWEEN       = 13               # seconds between requests
# ============================================================

TOPIC_TO_EVENT = {
    "earnings":                 "EARNINGS_REPORT",
    "ipo":                      "PRODUCT_LAUNCH",
    "mergers_and_acquisitions": "MERGER_ACQUISITION",
    "financial_markets":        "MACRO",
    "economy_macro":            "MACRO",
    "economy_monetary":         "FED_MONETARY_POLICY",
    "technology":               "PRODUCT_LAUNCH",
    "manufacturing":            "SECTOR_EVENT",
    "real_estate":              "SECTOR_EVENT",
    "retail_wholesale":         "SECTOR_EVENT",
    "energy_transportation":    "SECTOR_EVENT",
    "life_sciences":            "SECTOR_EVENT",
    "finance":                  "MACRO",
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def parse_if_string(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return []
    return val if val is not None else []


def get_target_relevance(ticker_sentiment, target_ticker):
    for item in parse_if_string(ticker_sentiment):
        if item.get("ticker") == target_ticker:
            return float(item.get("relevance_score", 0))
    return 0.0


def get_target_sentiment_score(ticker_sentiment, target_ticker):
    for item in parse_if_string(ticker_sentiment):
        if item.get("ticker") == target_ticker:
            return float(item.get("ticker_sentiment_score", 0))
    return 0.0


def get_event_label(topics):
    items = parse_if_string(topics)
    if not items:
        return "UNCLASSIFIABLE"
    top = sorted(items, key=lambda x: float(x.get("relevance_score", 0)), reverse=True)[0]
    return TOPIC_TO_EVENT.get(top.get("topic", ""), "UNCLASSIFIABLE")


def process_raw_df(df_raw, ticker):
    df = df_raw.copy()
    df["ticker"] = ticker

    df["time_published"] = pd.to_datetime(
        df["time_published"],
        format="%Y%m%dT%H%M%S",
        errors="coerce"
    )

    df["target_relevance"] = df["ticker_sentiment"].apply(
        lambda x: get_target_relevance(x, ticker)
    )
    df["target_sentiment_score"] = df["ticker_sentiment"].apply(
        lambda x: get_target_sentiment_score(x, ticker)
    )
    df["event_label"] = df["topics"].apply(get_event_label)

    df = df[df["target_relevance"] >= RELEVANCE_THRESHOLD].reset_index(drop=True)

    KEEP = [
        "ticker", "title", "summary", "source", "source_domain",
        "time_published", "overall_sentiment_score", "overall_sentiment_label",
        "target_relevance", "target_sentiment_score", "event_label", "url"
    ]
    return df[[c for c in KEEP if c in df.columns]]


# ------------------------------------------------------------
# Step 1 — Fetch
# ------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  BATCH : {TIME_FROM} → {TIME_TO}")
print(f"  TICKERS ({len(TICKERS)}): {TICKERS}")
print(f"{'='*55}\n")

if not API_KEY:
    raise EnvironmentError(
        "API key not found. Set it with: setx Vantage_api 'your_key' "
        "then restart your terminal/editor."
    )

frames = []

for i, ticker in enumerate(TICKERS, 1):
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
    data = response.json()

    if "feed" not in data:
        print(f"[{i}/{len(TICKERS)}] [{ticker}] WARN — no feed. Response: {data}")
        time.sleep(SLEEP_BETWEEN)
        continue

    df_raw = pd.DataFrame(data["feed"])
    df_processed = process_raw_df(df_raw, ticker)

    print(f"[{i}/{len(TICKERS)}] [{ticker}] Raw: {len(df_raw)} → Kept: {len(df_processed)}")
    frames.append(df_processed)
    time.sleep(SLEEP_BETWEEN)

if not frames:
    raise RuntimeError("Nothing fetched. Check API key, date range, or network.")

df_new = pd.concat(frames, ignore_index=True)
print(f"\n[BATCH TOTAL] {df_new.shape}")
print(f"[DATE RANGE ] {df_new['time_published'].min()} → {df_new['time_published'].max()}")

# ------------------------------------------------------------
# Step 2 — Append to existing CSV
# ------------------------------------------------------------
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

if SAVE_PATH.exists():
    df_existing = pd.read_csv(SAVE_PATH)
    rows_before = len(df_existing)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    print("[INFO] No existing file found. Creating fresh.")
    df_existing = pd.DataFrame()
    rows_before = 0
    df_combined = df_new

before_dedup = len(df_combined)
df_combined = df_combined.drop_duplicates(subset="url").reset_index(drop=True)
dupes_removed = before_dedup - len(df_combined)

df_combined.to_csv(SAVE_PATH, index=False)

# ------------------------------------------------------------
# Step 3 — Verification
# ------------------------------------------------------------
df_verify = pd.read_csv(SAVE_PATH)

print(f"\n{'='*55}")
print(f"  VERIFICATION")
print(f"{'='*55}")
print(f"  Rows before append     : {rows_before}")
print(f"  New rows fetched       : {len(df_new)}")
print(f"  Duplicates removed     : {dupes_removed}")
print(f"  Final rows in CSV      : {len(df_verify)}")
print(f"  Full date range        : {df_verify['time_published'].min()} → {df_verify['time_published'].max()}")
print(f"\n  Rows per ticker:")
print(df_verify["ticker"].value_counts().to_string())
print(f"\n  Event label distribution:")
print(df_verify["event_label"].value_counts().to_string())
print(f"\n  Null check:")
print(df_verify.isnull().sum().to_string())
print(f"{'='*55}\n")