import requests
import pandas as pd
import time
# import os
import ast
from pathlib import Path

API_KEY = '9K5BTZTGHH2M1P76'

# Keep to 10 tickers max for 2-day timeline
tickers = ["AAPL", "AMZN", "MSFT", "GOOGL", "NVDA",
           "META", "TSLA", "JPM", "V", "JNJ"]

# Event mapping from AV topics to your taxonomy
TOPIC_TO_EVENT = {
    "earnings": "EARNINGS_REPORT",
    "ipo":      "PRODUCT_LAUNCH",
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

def get_top_topic(topics_list):
    """Extract highest relevance topic and map to your taxonomy."""
    if not topics_list:
        return "UNCLASSIFIABLE"
    if isinstance(topics_list, str):
        topics_list = ast.literal_eval(topics_list)
    sorted_topics = sorted(
        topics_list,
        key=lambda x: float(x.get("relevance_score", 0)),
        reverse=True
    )
    top = sorted_topics[0]["topic"]
    return TOPIC_TO_EVENT.get(top, "UNCLASSIFIABLE")

def get_ticker_relevance(ticker_sentiment, target_ticker):
    """Get relevance score for the target ticker specifically."""
    if isinstance(ticker_sentiment, str):
        ticker_sentiment = ast.literal_eval(ticker_sentiment)
    for item in ticker_sentiment:
        if item["ticker"] == target_ticker:
            return float(item["relevance_score"])
    return 0.0

def get_ticker_sentiment_score(ticker_sentiment, target_ticker):
    if isinstance(ticker_sentiment, str):
        ticker_sentiment = ast.literal_eval(ticker_sentiment)
    for item in ticker_sentiment:
        if item["ticker"] == target_ticker:
            return float(item.get("ticker_sentiment_score", 0))
    return 0.0

Path("vantage").mkdir(exist_ok=True)
frames = []

for ticker in tickers:
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={ticker}"
        f"&limit=50"
        f"&apikey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "feed" not in data:
        print(f"No feed for {ticker}: {data}")
        time.sleep(15)
        continue

    news = data["feed"]
    df_temp = pd.DataFrame(news)
    df_temp["ticker"] = ticker
    frames.append(df_temp)
    print(f"{ticker}: {len(df_temp)} articles")
    time.sleep(15)  # stay safe on free tier

df = pd.concat(frames, ignore_index=True)

# Timestamp
df["time_published"] = pd.to_datetime(
    df["time_published"],
    format="%Y%m%dT%H%M%S",
    errors="coerce"
)

# Relevance filter — drop rows where article isn't really about your ticker
df["target_relevance"] = df.apply(
    lambda row: get_ticker_relevance(row["ticker_sentiment"], row["ticker"]),
    axis=1
)
df = df[df["target_relevance"] >= 0.6].reset_index(drop=True)

# Ticker-specific sentiment score
df["target_sentiment_score"] = df.apply(
    lambda row: get_ticker_sentiment_score(row["ticker_sentiment"], row["ticker"]),
    axis=1
)

# Weak event label from topics (your proxy label — not ground truth)
df["event_label"] = df["topics"].apply(get_top_topic)

# Keep only columns you actually need
KEEP_COLS = [
    "ticker", "title", "summary", "source", "source_domain",
    "time_published", "overall_sentiment_score", "overall_sentiment_label",
    "target_relevance", "target_sentiment_score", "event_label", "url"
]
df = df[[c for c in KEEP_COLS if c in df.columns]]

df.to_csv("vantage/alphavantage_clean.csv", index=False)
print(f"\nFinal shape: {df.shape}")
print(df["event_label"].value_counts())
print(df.head())