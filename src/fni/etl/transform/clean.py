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


def get_target_relevance(ticker_sentiment, target_ticker):     # get target relevance score
    for item in parse_if_string(ticker_sentiment):
        if item.get("ticker") == target_ticker:
            return float(item.get("relevance_score", 0))
    return 0.0


# def get_target_sentiment_score(ticker_sentiment, target_ticker):
#     for item in parse_if_string(ticker_sentiment):
#         if item.get("ticker") == target_ticker:
#             return float(item.get("ticker_sentiment_score", 0))
#     return 0.0

"""
def get_event_label(topics):
    items = parse_if_string(topics)
    if not items:
        return "UNCLASSIFIABLE"
    top = sorted(items, key=lambda x: float(x.get("relevance_score", 0)), reverse=True)[0]
    return TOPIC_TO_EVENT.get(top.get("topic", ""), "UNCLASSIFIABLE")
"""



def process_raw_df(df_raw, ticker):
    df = df_raw.copy()
    df["ticker"] = ticker

    df["time_published"] = pd.to_datetime(
        df["time_published"],
        format="%Y%m%dT%H%M%S",
        errors="coerce"
    )

    # df["target_relevance"] = df["ticker_sentiment"].apply(
    #     lambda x: get_target_relevance(x, ticker)
    # )
    # df["target_sentiment_score"] = df["ticker_sentiment"].apply(
    #     lambda x: get_target_sentiment_score(x, ticker)
    # )
    # df["event_label"] = df["topics"].apply(get_event_label)

    # df = df[df["target_relevance"] >= RELEVANCE_THRESHOLD].reset_index(drop=True)

    # KEEP = [
    #     "ticker", "title", "summary", "source", "source_domain",
    #     "time_published", "overall_sentiment_score", "overall_sentiment_label",
    #     "target_relevance", "target_sentiment_score", "event_label", "url"
    # ]
    # return df[[c for c in KEEP if c in df.columns]]