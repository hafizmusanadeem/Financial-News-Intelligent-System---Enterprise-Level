from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)

DEFAULT_NEWS_PATH = (
    Path(__file__).resolve().parents[5]
    / "artifacts" / "labelled" / "news_data_labeled.csv"
)

CATEGORY_WEIGHTS: dict[str, float] = {
    "Earnings":           1.00,
    "Fed Monetary Policy":0.95,
    "Merger/Acquisition": 0.90,
    "Regulatory/Legal":   0.80,
    "Economic Data":      0.75,
    "Market Movement":    0.65,
    "Leadership Change":  0.55,
    "Product Launch":     0.45,
    "Other":              0.20,
}

# Composite score weights — tunable
W_AR        = 0.50 # weight for abnormal return component
W_SENTIMENT = 0.25 # weight for sentiment component
W_CATEGORY  = 0.25 # weight for category component


class ImpactScoreCalculator:
    def __init__(self, news_path: str | None = None) -> None:
        self.news_path = Path(news_path) if news_path else DEFAULT_NEWS_PATH

    # ── Loaders ───────────────────────────────────────────────────────────

    def _load_news(self) -> pd.DataFrame:
        df = pd.read_csv(self.news_path)
        df["time_published"] = pd.to_datetime(
            df["time_published"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["ticker", "time_published"]).reset_index(drop=True)
        # Snap to effective trading day (weekend → next Monday)
        df["news_date"] = df["time_published"].dt.tz_localize(None).dt.normalize()
        df["effective_date"] = df["news_date"].apply(
            lambda d: d + BDay(0)
        )
        return df

    def _fetch_prices(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch OHLC for tickers + SPY (benchmark)."""
        all_tickers = sorted(set(tickers) | {"SPY"})
        frames: list[pd.DataFrame] = []

        for tkr in all_tickers:
            try:
                hist = yf.Ticker(tkr).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    actions=False,
                    auto_adjust=True,       # adjusted closes for accuracy
                )
                if hist.empty:
                    logger.warning(f"No price data for {tkr} — it will be excluded.")
                    continue
                hist = hist.reset_index()
                hist["ticker"] = tkr
                hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
                frames.append(hist[["ticker", "Date", "Close"]])
            except Exception as e:
                logger.warning(f"yfinance error for {tkr}: {e}")

        if not frames:
            raise RuntimeError("No price data fetched at all.")

        return pd.concat(frames, ignore_index=True)

    # ── Core computation ──────────────────────────────────────────────────

    def _build_close_lookup(
        self, price_df: pd.DataFrame
    ) -> dict[tuple[str, object], float]:
        price_df = price_df.copy()
        price_df["date_key"] = price_df["Date"].dt.date
        return price_df.set_index(["ticker", "date_key"])["Close"].to_dict()

    def _compute_abnormal_return(
        self,
        news_df: pd.DataFrame,
        close_lookup: dict[tuple[str, object], float],
    ) -> pd.DataFrame:
        df = news_df.copy()

        def _bday_date(d: pd.Timestamp, offset: int) -> object:
            return (d + BDay(offset)).date()

        df["day0_date"] = df["effective_date"].apply(lambda d: _bday_date(d, 0))
        df["day1_date"] = df["effective_date"].apply(lambda d: _bday_date(d, 1))

        df["day0_close_stock"] = df.apply(
            lambda r: close_lookup.get((r["ticker"], r["day0_date"])), axis=1
        )
        df["day1_close_stock"] = df.apply(
            lambda r: close_lookup.get((r["ticker"], r["day1_date"])), axis=1
        )
        df["day0_close_spy"] = df.apply(
            lambda r: close_lookup.get(("SPY", r["day0_date"])), axis=1
        )
        df["day1_close_spy"] = df.apply(
            lambda r: close_lookup.get(("SPY", r["day1_date"])), axis=1
        )

        missing_before = len(df)
        df = df.dropna(
            subset=["day0_close_stock", "day1_close_stock",
                    "day0_close_spy",   "day1_close_spy"]
        ).copy()
        logger.info(
            f"Rows with full price data: {len(df)}/{missing_before} "
            f"({missing_before - len(df)} dropped — boundary/delisted)"
        )

        df["stock_return"] = (
            (df["day1_close_stock"] - df["day0_close_stock"])
            / df["day0_close_stock"]
        ).round(6)

        df["spy_return"] = (
            (df["day1_close_spy"] - df["day0_close_spy"])
            / df["day0_close_spy"]
        ).round(6)

        # Market-adjusted abnormal return
        df["abnormal_return"] = (df["stock_return"] - df["spy_return"]).round(6)

        return df

    def _compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Component 1: AR score (clipped at 5% move = max score 1.0) ───
        df["ar_score"] = df["abnormal_return"].abs().clip(upper=0.05) / 0.05

        # ── Component 2: Sentiment magnitude ─────────────────────────────
        if "overall_sentiment_score" in df.columns:
            df["sentiment_score"] = df["overall_sentiment_score"].abs().clip(upper=1.0)
        else:
            logger.warning("overall_sentiment_score missing — defaulting to 0.5")
            df["sentiment_score"] = 0.5

        # ── Component 3: Category weight ──────────────────────────────────
        if "event_label" in df.columns:
            df["category_score"] = (
                df["event_label"]
                .map(CATEGORY_WEIGHTS)
                .fillna(CATEGORY_WEIGHTS["Other"])
            )
        else:
            logger.warning("event_label missing — defaulting category_score to 0.5")
            df["category_score"] = 0.5

        # ── Composite ────────────────────────────────────────────────────
        df["impact_score"] = (
            W_AR        * df["ar_score"]
            + W_SENTIMENT * df["sentiment_score"]
            + W_CATEGORY  * df["category_score"]
        ).round(4)

        # ── Impact tier ──────────────────────────────────────────────────
        df["impact_tier"] = pd.cut(
            df["impact_score"],
            bins=[0.0, 0.35, 0.55, 0.75, 1.01],
            labels=["Low", "Medium", "High", "Critical"],
            right=False,
        )

        # ── Direction ─────────────────────────────────────────────────────
        df["impact_direction"] = np.where(
            df["abnormal_return"] > 0, "Positive",
            np.where(df["abnormal_return"] < 0, "Negative", "Neutral")
        )

        return df

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, output_path: str | None = None) -> pd.DataFrame:
        news_df = self._load_news()
        tickers = sorted(news_df["ticker"].dropna().unique().tolist())

        min_date = news_df["effective_date"].min()
        max_date = news_df["effective_date"].max()

        price_df = self._fetch_prices(
            tickers=tickers,
            start=min_date - pd.Timedelta(days=10),
            end=max_date   + pd.Timedelta(days=10),
        )

        close_lookup = self._build_close_lookup(price_df)
        df = self._compute_abnormal_return(news_df, close_lookup)
        df = self._compute_composite_score(df)

        output_cols = [
            "ticker", "time_published", "news_date", "effective_date",
            "day0_date", "day1_date",
            "day0_close_stock", "day1_close_stock",
            "stock_return", "spy_return", "abnormal_return",
            "ar_score", "sentiment_score", "category_score",
            "impact_score", "impact_tier", "impact_direction",
            "event_label", "title", "summary", "source",
            "overall_sentiment_label",
        ]
        final_cols = [c for c in output_cols if c in df.columns]
        result = df[final_cols].sort_values("impact_score", ascending=False).reset_index(drop=True)

        logger.info(f"\nImpact tier distribution:\n{result['impact_tier'].value_counts().to_string()}")
        logger.info(f"\nTop 5 by impact:\n{result[['ticker','title','impact_score','impact_tier']].head().to_string()}")

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False)
            logger.info(f"Saved to {out}")

        return result


def main() -> None:
    calc = ImpactScoreCalculator()
    output_path = (
        Path(__file__).resolve().parents[5]
        / "artifacts" / "transform" / "impact_score"
        / "impact_scores.csv"
    )
    result = calc.run(output_path=str(output_path))
    print(f"\nProcessed {len(result)} rows | {result['ticker'].nunique()} tickers")


if __name__ == "__main__":
    main()