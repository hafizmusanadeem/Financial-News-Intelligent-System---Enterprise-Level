"""
src/fni/etl/pipeline.py

Single entry-point that runs the full ETL pipeline end-to-end:
    1. Extract  — AlphaVantage news ingestion
    2. Clean    — drop nulls, strip whitespace, dedup
    3. Label    — LLM event classification
    4. Score    — next-day price impact computation
    5. Load     — PostgreSQL upsert

Each stage is independently skip-able for partial re-runs and debugging.

Run:
    python -m src.fni.etl.pipeline
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException

# BUG FIX: TICKERS was not in constants.py in the original pipeline.
# It has been moved there from alphavantage.py.
from src.fni.core.constants import (
    TICKERS,
    TIME_FROM,
    TIME_TO,
    SLEEP_BETWEEN,
    LIMIT,
    REQUEST_DELAY,
)
from src.fni.etl.extract_from_sources.alphavantage import AlphaVantageExtractor
from src.fni.etl.transform.clean import DataCleaner
from src.fni.etl.transform.labeller_LLM.label import EventLabeller
from src.fni.etl.transform.impact_score.get_impact_score import ImpactScoreDataLoader
from src.fni.etl.load.load import NewsLoader
from src.fni.etl.load.configuration import get_db_connection

setup_logger()
logger = get_logger(__name__)


# ── Artifact paths ─────────────────────────────────────────────────────────────

_ROOT: Path = Path(__file__).resolve().parents[3]

INTERIM_PATH: Path  = _ROOT / "artifacts" / "interim"                        / "news_data.csv"
CLEAN_PATH: Path    = _ROOT / "artifacts" / "interim"                        / "news_data_clean.csv"
LABELLED_PATH: Path = _ROOT / "artifacts" / "labelled"                       / "news_data_labeled.csv"
IMPACT_PATH: Path   = _ROOT / "artifacts" / "transform" / "impact_score"     / "news_with_impact.csv"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    tickers:       list[str] = field(default_factory=lambda: TICKERS)
    time_from:     str       = TIME_FROM
    time_to:       str       = TIME_TO
    limit:         int       = LIMIT
    sleep_between: int       = SLEEP_BETWEEN
    request_delay: int       = REQUEST_DELAY
    skip_extract:  bool      = False
    skip_clean:    bool      = False
    skip_label:    bool      = False
    skip_score:    bool      = False
    skip_load:     bool      = False   # set True for dry runs


# ── Stage runners ──────────────────────────────────────────────────────────────

def run_extract(cfg: PipelineConfig) -> None:
    """Stage 1: Pull raw news from AlphaVantage → artifacts/interim/news_data.csv"""
    if cfg.skip_extract:
        if not INTERIM_PATH.exists():
            raise CustomException(
                f"skip_extract=True but interim file not found: {INTERIM_PATH}", sys
            )
        logger.info(f"[EXTRACT] Skipped — reusing {INTERIM_PATH}")
        return

    logger.info("[EXTRACT] Starting AlphaVantage ingestion ...")
    AlphaVantageExtractor(
        tickers=cfg.tickers,
        time_from=cfg.time_from,
        time_to=cfg.time_to,
        save_path=str(INTERIM_PATH),
        limit=cfg.limit,
        sleep_between=cfg.sleep_between,
    ).run()
    logger.info(f"[EXTRACT] Done → {INTERIM_PATH}")


def run_clean(cfg: PipelineConfig) -> None:
    """Stage 2: Drop nulls, strip whitespace, dedup → artifacts/interim/news_data_clean.csv"""
    if cfg.skip_clean:
        if not CLEAN_PATH.exists():
            # Fall back to raw interim if clean file doesn't exist yet
            logger.warning(
                "[CLEAN] Skipped but clean file not found — "
                "downstream stages will use raw interim CSV."
            )
        else:
            logger.info(f"[CLEAN] Skipped — reusing {CLEAN_PATH}")
        return

    if not INTERIM_PATH.exists():
        raise CustomException(
            f"[CLEAN] Cannot run — interim file missing: {INTERIM_PATH}. "
            "Run extract stage first.", sys
        )

    logger.info("[CLEAN] Starting data cleaning ...")
    DataCleaner(
        input_path=str(INTERIM_PATH),
        output_path=str(CLEAN_PATH),
    ).run()
    logger.info(f"[CLEAN] Done → {CLEAN_PATH}")


def run_label(cfg: PipelineConfig) -> None:
    """Stage 3: LLM event classification → artifacts/labelled/news_data_labeled.csv"""
    if cfg.skip_label:
        if not LABELLED_PATH.exists():
            raise CustomException(
                f"skip_label=True but labelled file not found: {LABELLED_PATH}", sys
            )
        logger.info(f"[LABEL] Skipped — reusing {LABELLED_PATH}")
        return

    # Use cleaned file if available, else fall back to raw interim
    label_input = CLEAN_PATH if CLEAN_PATH.exists() else INTERIM_PATH
    if not label_input.exists():
        raise CustomException(
            f"[LABEL] Cannot run — no input file found at {label_input}. "
            "Run extract/clean stages first.", sys
        )

    logger.info(f"[LABEL] Starting LLM labelling from {label_input} ...")
    EventLabeller(
        input_path=str(label_input),
        output_path=str(LABELLED_PATH),
        request_delay=cfg.request_delay,
    ).run()
    # NOTE: EventLabeller.run() no longer calls sys.exit(0) on completion.
    # It returns gracefully even when all rows are already labeled.
    logger.info(f"[LABEL] Done → {LABELLED_PATH}")


def run_impact_score(cfg: PipelineConfig) -> pd.DataFrame:
    """Stage 4: Next-day price impact → artifacts/transform/impact_score/"""
    if cfg.skip_score:
        if not IMPACT_PATH.exists():
            raise CustomException(
                f"skip_score=True but impact file not found: {IMPACT_PATH}", sys
            )
        logger.info(f"[SCORE] Skipped — loading from {IMPACT_PATH}")
        return pd.read_csv(IMPACT_PATH)

    if not LABELLED_PATH.exists():
        raise CustomException(
            f"[SCORE] Cannot run — labelled file missing: {LABELLED_PATH}. "
            "Run label stage first.", sys
        )

    logger.info("[SCORE] Computing next-day impact scores ...")
    df: pd.DataFrame = ImpactScoreDataLoader(
        news_path=str(LABELLED_PATH)
    ).run(output_path=str(IMPACT_PATH))
    logger.info(f"[SCORE] Done → {IMPACT_PATH} | rows: {len(df)}")
    return df


def run_load(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Stage 5: Upsert final dataset into PostgreSQL."""
    if cfg.skip_load:
        logger.info("[LOAD] Skipped (dry run mode).")
        return

    if df.empty:
        logger.warning("[LOAD] Empty DataFrame — nothing to load.")
        return

    logger.info(f"[LOAD] Loading {len(df)} rows into PostgreSQL ...")
    conn = get_db_connection()
    loader = NewsLoader(conn=conn)
    try:
        loader.load(df)
    finally:
        loader.close()
    logger.info("[LOAD] Done.")


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig | None = None) -> None:
    """
    Runs the full ETL pipeline sequentially.
    Raises CustomException on any stage failure (no silent swallowing).
    """
    if cfg is None:
        cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("FNI ETL Pipeline — START")
    logger.info(f"  Range   : {cfg.time_from} → {cfg.time_to}")
    logger.info(f"  Tickers : {len(cfg.tickers)}")
    logger.info(
        f"  Stages  : extract={not cfg.skip_extract} | clean={not cfg.skip_clean} | "
        f"label={not cfg.skip_label} | score={not cfg.skip_score} | "
        f"load={not cfg.skip_load}"
    )
    logger.info("=" * 60)

    try:
        run_extract(cfg)
        run_clean(cfg)
        run_label(cfg)
        df_final: pd.DataFrame = run_impact_score(cfg)
        run_load(df_final, cfg)
    except CustomException:
        logger.error("Pipeline aborted. See traceback above.")
        raise
    except Exception as e:
        raise CustomException(f"Unexpected pipeline error: {e}", sys) from e

    logger.info("=" * 60)
    logger.info("FNI ETL Pipeline — COMPLETE")
    logger.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(PipelineConfig(
        skip_extract=False,
        skip_clean=False,
        skip_label=False,
        skip_score=False,
        skip_load=False,
    ))