"""
src/fni/etl/transform/impact_score/get_impact_score.py

Joins labelled news data with Yahoo Finance OHLC prices to compute:
    - next_day_return  : (day1_close - day0_close) / day0_close
    - impact_score     : normalised [0, 1] magnitude (capped at 5 % move = 1.0)
    - impact_label     : Very High / High / Medium / Low

day0 = the calendar date the news was published
day1 = the next trading day after day0
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException

setup_logger()
logger = get_logger(__name__)

DEFAULT_NEWS_PATH: Path = (
    Path(__file__).resolve().parents[5]
    / "artifacts"
    / "labelled"
    / "news_data_labeled.csv"
)


class ImpactScoreDataLoader:

    def __init__(self, news_path: str | None = None) -> None:
        self.news_path: Path = (
            Path(news_path) if news_path else DEFAULT_NEWS_PATH
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_news(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.news_path)
        except Exception as e:
            raise CustomException(
                f"Failed to load labelled news CSV: {e}", sys
            ) from e

        if "time_published" not in df.columns:
            raise CustomException(
                "Expected a 'time_published' column in the news dataset.", sys
            )

        df["time_published"] = pd.to_datetime(
            df["time_published"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["ticker", "time_published"]).reset_index(drop=True)
        df["news_date"] = (
            df["time_published"].dt.normalize().dt.tz_localize(None).dt.date
        )
        logger.info(f"[SCORE] Loaded {len(df)} labelled news rows.")
        return df

    def _get_news_tickers(self, df: pd.DataFrame) -> list[str]:
        return sorted(df["ticker"].dropna().astype(str).unique().tolist())

    def _fetch_price_history(
        self,
        tickers: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            try:
                hist: pd.DataFrame = yf.Ticker(ticker).history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    actions=False,
                    auto_adjust=False,
                )
            except Exception as e:
                logger.warning(f"[SCORE] yfinance failed for {ticker}: {e} — skipping.")
                continue

            if hist.empty:
                logger.warning(f"[SCORE] No price data returned for {ticker} — skipping.")
                continue

            hist = hist.reset_index()
            hist["ticker"] = ticker
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
            hist = hist[["ticker", "Date", "Open", "High", "Low", "Close"]]
            frames.append(hist)

        if not frames:
            logger.warning("[SCORE] No price data fetched for any ticker.")
            return pd.DataFrame(columns=["ticker", "Date", "Open", "High", "Low", "Close"])

        result = pd.concat(frames, ignore_index=True)
        logger.info(f"[SCORE] Fetched price history — {len(result)} rows across {len(tickers)} tickers.")
        return result

    def _align_next_day_news(
        self,
        news_df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if price_df.empty:
            logger.warning("[SCORE] Price DataFrame is empty — cannot compute impact scores.")
            return pd.DataFrame()

        price_df = price_df.copy()
        price_df["price_date"] = price_df["Date"].dt.date
        price_df["next_news_date"] = (
            price_df["Date"] + BDay(1)
        ).dt.normalize().dt.date

        # day0_close: close price on the news publication date
        day0_prices = (
            price_df[["ticker", "price_date", "Close"]]
            .rename(columns={"Close": "day0_close", "price_date": "news_date"})
        )

        # Merge day1 prices: the trading day after news was published
        merged = news_df.merge(
            price_df[[
                "ticker", "price_date", "next_news_date",
                "Open", "High", "Low", "Close",
            ]],
            left_on=["ticker", "news_date"],
            right_on=["ticker", "next_news_date"],
            how="inner",
        ).rename(columns={"Close": "day1_close"})

        # Attach day0 close
        merged = merged.merge(day0_prices, on=["ticker", "news_date"], how="left")

        # Compute metrics
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
        selected_columns: list[str] = [
            "ticker", "time_published", "news_date", "price_date",
            "day0_close", "day1_close", "next_day_return",
            "impact_score", "impact_label",
            "Open", "High", "Low",
        ] + other_news_columns

        result = merged[selected_columns].drop_duplicates().reset_index(drop=True)

        # Diagnostic logging (was raw print() calls — fixed)
        logger.info(
            f"[SCORE] news_rows={len(news_df)} | price_rows={len(price_df)} "
            f"| after_merge={len(result)}"
        )
        news_counts = news_df["ticker"].value_counts()
        aligned_counts = result["ticker"].value_counts()
        ticker_summary = (
            pd.DataFrame({"news": news_counts, "aligned": aligned_counts})
            .fillna(0)
            .astype(int)
        )
        logger.info(f"[SCORE] Per-ticker alignment:\n{ticker_summary.to_string()}")

        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, output_path: str | None = None) -> pd.DataFrame:
        news_df = self._load_news()
        tickers = self._get_news_tickers(news_df)

        if not tickers:
            raise CustomException(
                "No tickers found in the news dataset.", sys
            )

        min_date = news_df["time_published"].dt.date.min()
        max_date = news_df["time_published"].dt.date.max()

        price_df = self._fetch_price_history(
            tickers=tickers,
            start_date=pd.Timestamp(min_date) - pd.Timedelta(days=7),
            end_date=pd.Timestamp(max_date) + pd.Timedelta(days=7),
        )

        aligned_df = self._align_next_day_news(news_df, price_df)

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            aligned_df.to_csv(out, index=False)
            logger.info(f"[SCORE] Saved {len(aligned_df)} rows → {out}")

        return aligned_df


def main() -> None:
    loader = ImpactScoreDataLoader()
    output: Path = (
        Path(__file__).resolve().parents[5]
        / "artifacts" / "transform" / "impact_score"
        / "next_day_news_with_stock_prices.csv"
    )
    result = loader.run(output_path=str(output))
    logger.info(
        f"Extracted {len(result)} next-day rows "
        f"for {result['ticker'].nunique()} tickers."
    )


if __name__ == "__main__":
    main()