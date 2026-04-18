# import finnhub
# import pandas as pd

# API_KEY = 'd7gb74pr01qqb8rj0kp0d7gb74pr01qqb8rj0kpg'

# client = finnhub.Client(api_key=API_KEY)

# tickers = ['AAPL', 'AMZN', 'MSFT']

# frames = []

# for ticker in tickers:
#     news = client.company_news(
#         ticker,
#         _from="2026-04-10",
#         to="2026-04-17"
#     )

#     df_temp = pd.DataFrame(news)
#     df_temp["ticker"] = ticker
#     frames.append(df_temp)

# df = pd.concat(frames, ignore_index=True)

# df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce")

# df.to_csv("finnhub_data.csv", index=False)

# print(df.head())


# Vantage Api
import requests
import pandas as pd
import time

API_KEY = '9K5BTZTGHH2M1P76'

tickers = ["AAPL", "AMZN", "MSFT"]
frames = []

for ticker in tickers:
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={ticker}"
        f"&apikey={API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    news = data.get("feed", [])

    df_temp = pd.DataFrame(news)
    df_temp["ticker"] = ticker

    frames.append(df_temp)

    time.sleep(12)   # Alpha Vantage free-tier rate limit safety

df = pd.concat(frames, ignore_index=True)

# convert timestamp
df["time_published"] = pd.to_datetime(
    df["time_published"],
    format="%Y%m%dT%H%M%S",
    errors="coerce"
)

df.to_csv("vantage/alphavantage_news.csv", index=False)

print(df.head())