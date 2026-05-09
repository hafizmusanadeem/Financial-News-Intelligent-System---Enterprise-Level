from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


DEFAULT_NEWS_PATH = (
    Path(__file__).resolve().parents[5]
    / "artifacts"
    / "labelled"
    / "news_data_labeled.csv"
)


class ImpactScoreDataLoader:
    def __init__(self, news_path: str | None = None):
        self.news_path = Path(news_path) if news_path else DEFAULT_NEWS_PATH

    def _load_news(self) -> pd.DataFrame:
        df = pd.read_csv(self.news_path)
        if "time_published" not in df.columns:
            raise ValueError("Expected a time_published column in the news dataset.")

        df["time_published"] = pd.to_datetime(df["time_published"], utc=True, errors="coerce")
        df = df.dropna(subset=["ticker", "time_published"]).reset_index(drop=True)
        df["news_date"] = df["time_published"].dt.normalize().dt.tz_localize(None).dt.date
        return df

    def _get_news_tickers(self, df: pd.DataFrame) -> list[str]:
        return sorted(df["ticker"].dropna().astype(str).unique().tolist())

    def _fetch_price_history(
        self,
        tickers: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            hist = yf.Ticker(ticker).history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                actions=False,
                auto_adjust=False,
            )
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist["ticker"] = ticker

            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None) # Ensure Date is timezone-naive for merging
            hist = hist[["ticker", "Date", "Open", "High", "Low", "Close"]]
            frames.append(hist)

        if not frames:
            return pd.DataFrame(columns=["ticker", "Date", "Open", "High", "Low", "Close"])

        return pd.concat(frames, ignore_index=True)

    def _align_next_day_news(
        self, news_df: pd.DataFrame, 
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        if price_df.empty:
            return pd.DataFrame()

        price_df = price_df.copy()
        price_df["price_date"] = price_df["Date"].dt.date

        # next_news_date = the trading day AFTER this price row
        price_df["next_news_date"] = (
            price_df["Date"] + BDay(1)
        ).dt.normalize().dt.date

        # Bring in day 0 close — the price on the same date as the news
        # This is needed to compute next_day_return = (day1_close - day0_close) / day0_close
        day0_prices = price_df[["ticker", "price_date", "Close"]].rename(
            columns={"Close": "day0_close", "price_date": "news_date"}
        )

        # Merge day 1 prices (next trading day after news)
        merged = news_df.merge(
            price_df[["ticker", "price_date", "next_news_date", "Open", "High", "Low", "Close"]],
            left_on=["ticker", "news_date"],
            right_on=["ticker", "next_news_date"],
            how="inner",
        ).rename(columns={"Close": "day1_close"})

        # Merge day 0 close in
        merged = merged.merge(
            day0_prices,
            on=["ticker", "news_date"],
            how="left"
        )

        # Compute next_day_return and impact columns
        merged["next_day_return"] = (
            (merged["day1_close"] - merged["day0_close"]) / merged["day0_close"]
        ).round(4)

        merged["impact_score"] = merged["next_day_return"].apply(
            lambda x: round(min(abs(x) / 0.05, 1.0), 4) if pd.notnull(x) else None
        )

        merged["impact_label"] = merged["next_day_return"].apply(
            lambda x: (
                "Very High" if abs(x) >= 0.05 else
                "High"      if abs(x) >= 0.02 else
                "Medium"    if abs(x) >= 0.01 else
                "Low"
            ) if pd.notnull(x) else None
        )

        other_news_columns = [
            c for c in news_df.columns
            if c not in {"ticker", "time_published", "news_date"}
        ]
        selected_columns = [
            "ticker", "time_published", "news_date", "price_date",
            "day0_close", "day1_close", "next_day_return",
            "impact_score", "impact_label",
            "Open", "High", "Low",
        ] + other_news_columns

        return merged[selected_columns].drop_duplicates().reset_index(drop=True)

    def run(self, output_path: str | None = None) -> pd.DataFrame:
        news_df = self._load_news()
        tickers = self._get_news_tickers(news_df)
        if not tickers:
            raise ValueError("No tickers found in the news dataset.")

        min_date = news_df["time_published"].dt.date.min()
        max_date = news_df["time_published"].dt.date.max()

        price_df = self._fetch_price_history(
            tickers=tickers,
            start_date=pd.Timestamp(min_date - pd.Timedelta(days=7)),
            end_date=pd.Timestamp(max_date + pd.Timedelta(days=7)),
        )

        aligned_df = self._align_next_day_news(news_df, price_df)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            aligned_df.to_csv(output_file, index=False)

        # Add this temporarily inside run() after the merge
        print(f"News rows: {len(news_df)}")
        print(f"Price rows: {len(price_df)}")
        print(f"After merge: {len(aligned_df)}")

        # Check which tickers are losing the most rows
        news_counts = news_df["ticker"].value_counts()
        aligned_counts = aligned_df["ticker"].value_counts()
        print(pd.DataFrame({"news": news_counts, "aligned": aligned_counts}).fillna(0).astype(int))


        return aligned_df


def main() -> None:
    loader = ImpactScoreDataLoader()
    output_path = (
        Path(__file__).resolve().parents[5]
        / "artifacts"
        / "transform"
        / "impact_score"
        / "next_day_news_with_stock_prices.csv"
    )
    result = loader.run(output_path=str(output_path))
    print(f"Extracted {len(result)} next-day news rows for {len(result['ticker'].unique())} tickers.")
    print(f"Saved aligned dataset to {output_path}")


if __name__ == "__main__":
    main()
