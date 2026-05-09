from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException

setup_logger()
logger = get_logger(__name__)

# ── Valid value sets (source of truth for downstream DB constraints) ──────────

VALID_EVENT_LABELS: frozenset[str] = frozenset({
    "Earnings", "Merger/Acquisition", "Product Launch",
    "Regulatory/Legal", "Market Movement", "Leadership Change",
    "Economic Data", "Fed Monetary Policy", "Other",
})

VALID_IMPACT_TIERS: frozenset[str] = frozenset({
    "Low", "Medium", "High", "Critical",
})

VALID_IMPACT_DIRECTIONS: frozenset[str] = frozenset({
    "Positive", "Negative", "Neutral",
})

VALID_SENTIMENT_LABELS: frozenset[str] = frozenset({
    "Bullish", "Somewhat-Bullish", "Neutral",
    "Somewhat-Bearish", "Bearish",
})

# ── Expected schema — col: (pandas_dtype, nullable) ──────────────────────────

SCHEMA: dict[str, tuple[str, bool]] = {
    "ticker":                   ("object",   False),
    "time_published":           ("datetime", False),
    "news_date":                ("date",     False),
    "effective_date":           ("date",     False),
    "day0_date":                ("date",     False),
    "day1_date":                ("date",     False),
    "day0_close_stock":         ("float64",  False),
    "day1_close_stock":         ("float64",  False),
    "stock_return":             ("float64",  False),
    "spy_return":               ("float64",  False),
    "abnormal_return":          ("float64",  False),
    "ar_score":                 ("float64",  False),
    "sentiment_score":          ("float64",  False),
    "category_score":           ("float64",  False),
    "impact_score":             ("float64",  False),
    "impact_tier":              ("category", False),
    "impact_direction":         ("category", False),
    "event_label":              ("category", False),
    "title":                    ("object",   False),
    "summary":                  ("object",   False),
    "source":                   ("object",   False),
    "overall_sentiment_label":  ("category", False),
}


class DataCleaner:
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path  = Path(input_path)
        self.output_path = Path(output_path)

    # ── Steps ─────────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded — shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(f"Failed to load: {e}", sys)

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        missing = set(SCHEMA.keys()) - set(df.columns)
        if missing:
            raise CustomException(f"Missing required columns: {missing}", sys)

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Timestamps
            df["time_published"] = pd.to_datetime(
                df["time_published"], utc=True, errors="raise"
            )

            # Date-only columns — cast from string/datetime → date
            for col in ("news_date", "effective_date", "day0_date", "day1_date"):
                df[col] = pd.to_datetime(df[col], errors="raise").dt.date

            # Categoricals
            for col in ("impact_tier", "impact_direction", "event_label",
                        "overall_sentiment_label"):
                df[col] = df[col].astype("category")

            # Floats — enforce precision (6 dp is sufficient for returns)
            for col in ("day0_close_stock", "day1_close_stock", "stock_return",
                        "spy_return", "abnormal_return", "ar_score",
                        "sentiment_score", "category_score", "impact_score"):
                df[col] = df[col].round(6).astype("float64")

            logger.info("Type casting complete.")
            return df

        except Exception as e:
            raise CustomException(f"Type casting failed: {e}", sys)

    def _validate_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        checks: dict[str, frozenset[str]] = {
            "event_label":             VALID_EVENT_LABELS,
            "impact_tier":             VALID_IMPACT_TIERS,
            "impact_direction":        VALID_IMPACT_DIRECTIONS,
            "overall_sentiment_label": VALID_SENTIMENT_LABELS,
        }

        for col, valid_set in checks.items():
            actual = set(df[col].astype(str).unique())
            invalid = actual - valid_set
            if invalid:
                # Log and quarantine — do NOT silently drop
                bad_rows = df[df[col].astype(str).isin(invalid)]
                logger.warning(
                    f"Column '{col}' has {len(bad_rows)} rows with invalid values: "
                    f"{invalid} — these rows will be quarantined."
                )
                df = df[~df[col].astype(str).isin(invalid)].copy()

        logger.info("Categorical validation complete.")
        return df

    def _validate_numeric_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        # Scores must be in [0, 1]
        for col in ("ar_score", "sentiment_score", "category_score", "impact_score"):
            out_of_range = df[(df[col] < 0.0) | (df[col] > 1.0)]
            if not out_of_range.empty:
                raise CustomException(
                    f"Column '{col}' has {len(out_of_range)} values outside [0, 1]. "
                    f"Fix upstream in impact score calculation.", sys
                )

        # Date ordering: day0 must precede day1
        day0 = pd.to_datetime(df["day0_date"].astype(str))
        day1 = pd.to_datetime(df["day1_date"].astype(str))
        if (day0 >= day1).any():
            raise CustomException(
                "day0_date >= day1_date detected — price alignment is broken.", sys
            )

        logger.info("Numeric range validation complete.")
        return df

    def _enforce_nullability(self, df: pd.DataFrame) -> pd.DataFrame:
        non_nullable = [col for col, (_, nullable) in SCHEMA.items() if not nullable]
        null_counts = df[non_nullable].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if not cols_with_nulls.empty:
            raise CustomException(
                f"Unexpected nulls in non-nullable columns:\n{cols_with_nulls.to_string()}",
                sys,
            )

        logger.info("Null enforcement complete.")
        return df

    def _save(self, df: pd.DataFrame) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Save with consistent date format for downstream loader
            df.to_csv(self.output_path, index=False)
            logger.info(
                f"Clean dataset saved — shape: {df.shape} | path: {self.output_path}"
            )
        except Exception as e:
            raise CustomException(f"Failed to save clean output: {e}", sys)

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        df = self._load()
        self._check_required_columns(df)
        df = self._cast_types(df)
        df = self._validate_categoricals(df)
        df = self._validate_numeric_ranges(df)
        df = self._enforce_nullability(df)
        self._save(df)

        logger.info(
            f"Clean run complete — {len(df)} rows ready for load.\n"
            f"Impact tier distribution:\n"
            f"{df['impact_tier'].value_counts().to_string()}"
        )
        return df