"""
src/fni/etl/pipeline.py

Single entry-point that runs the full ETL pipeline:
    1. Extract  — AlphaVantage news ingestion
    2. Label    — LLM-based event classification
    3. Score    — Next-day price impact computation
    4. Load     — PostgreSQL upsert

Run:
    python -m src.fni.etl.pipeline
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException
from src.fni.core.constants import TICKERS, TIME_FROM, TIME_TO, SLEEP_BETWEEN, LIMIT
from src.fni.etl.extract_from_sources.alphavantage import AlphaVantageExtractor
from src.fni.etl.transform.labeller_LLM.label import EventLabeller
from src.fni.etl.transform.impact_score.get_impact_score import ImpactScoreDataLoader
from src.fni.etl.load.load import NewsLoader
from src.fni.etl.load.configuration import get_db_connection

setup_logger()
logger = get_logger(__name__)

# ── Artifact paths (single source of truth for the whole pipeline) ─────────────

_BASE: Path = Path(__file__).resolve().parents[3]  # project root

INTERIM_PATH: Path   = _BASE / "artifacts" / "interim"   / "news_data.csv"
LABELLED_PATH: Path  = _BASE / "artifacts" / "labelled"  / "news_data_labeled.csv"
IMPACT_PATH: Path    = _BASE / "artifacts" / "transform"  / "impact_score" / "news_with_impact.csv"


# ── Pipeline config ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    tickers:        list[str] = field(default_factory=lambda: TICKERS)
    time_from:      str       = TIME_FROM
    time_to:        str       = TIME_TO
    limit:          int       = LIMIT
    sleep_between:  int       = SLEEP_BETWEEN
    request_delay:  int       = 3          # seconds between LLM calls
    skip_extract:   bool      = False      # set True to reuse existing interim CSV
    skip_label:     bool      = False      # set True to reuse existing labelled CSV
    skip_score:     bool      = False      # set True to reuse existing impact CSV
    skip_load:      bool      = False      # set True to skip DB load (dry run)


# ── Stage runners ───────────────────────────────────────────────────────────────

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
    extractor = AlphaVantageExtractor(
        tickers=cfg.tickers,
        time_from=cfg.time_from,
        time_to=cfg.time_to,
        save_path=str(INTERIM_PATH),
        limit=cfg.limit,
        sleep_between=cfg.sleep_between,
    )
    extractor.run()
    logger.info(f"[EXTRACT] Done → {INTERIM_PATH}")


def run_label(cfg: PipelineConfig) -> None:
    """Stage 2: LLM event classification → artifacts/labelled/news_data_labeled.csv"""
    if cfg.skip_label:
        if not LABELLED_PATH.exists():
            raise CustomException(
                f"skip_label=True but labelled file not found: {LABELLED_PATH}", sys
            )
        logger.info(f"[LABEL] Skipped — reusing {LABELLED_PATH}")
        return

    if not INTERIM_PATH.exists():
        raise CustomException(
            f"[LABEL] Cannot run — interim file missing: {INTERIM_PATH}. "
            "Run extract stage first.", sys
        )

    logger.info("[LABEL] Starting LLM event labelling ...")
    labeller = EventLabeller(
        input_path=str(INTERIM_PATH),
        output_path=str(LABELLED_PATH),
        request_delay=cfg.request_delay,
    )
    labeller.run()
    logger.info(f"[LABEL] Done → {LABELLED_PATH}")


def run_impact_score(cfg: PipelineConfig) -> pd.DataFrame:
    """Stage 3: Compute next-day price impact → artifacts/transform/impact_score/"""
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
    scorer = ImpactScoreDataLoader(news_path=str(LABELLED_PATH))
    df: pd.DataFrame = scorer.run(output_path=str(IMPACT_PATH))
    logger.info(f"[SCORE] Done → {IMPACT_PATH} | rows: {len(df)}")
    return df


def run_load(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Stage 4: Upsert final dataset into PostgreSQL."""
    if cfg.skip_load:
        logger.info("[LOAD] Skipped (dry run mode).")
        return

    if df.empty:
        logger.warning("[LOAD] Empty DataFrame — nothing to load into PostgreSQL.")
        return

    logger.info(f"[LOAD] Loading {len(df)} rows into PostgreSQL ...")
    conn = get_db_connection()
    loader = NewsLoader(conn=conn)
    try:
        loader.load(df)
    finally:
        loader.close()
    logger.info("[LOAD] Done.")


# ── Orchestrator ────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig | None = None) -> None:
    """
    Runs the full ETL pipeline sequentially.
    Each stage is independently skip-able for re-runs and debugging.
    """
    if cfg is None:
        cfg = PipelineConfig()

    logger.info("=" * 60)
    logger.info("FNI ETL Pipeline — START")
    logger.info(f"  Time range : {cfg.time_from} → {cfg.time_to}")
    logger.info(f"  Tickers    : {len(cfg.tickers)}")
    logger.info(f"  Stages     : extract={not cfg.skip_extract} | label={not cfg.skip_label} | "
                f"score={not cfg.skip_score} | load={not cfg.skip_load}")
    logger.info("=" * 60)

    try:
        run_extract(cfg)
        run_label(cfg)
        df_final: pd.DataFrame = run_impact_score(cfg)
        run_load(df_final, cfg)
    except CustomException:
        logger.error("Pipeline aborted due to a CustomException. Check logs above.")
        raise
    except Exception as e:
        raise CustomException(f"Unexpected pipeline error: {e}", sys) from e

    logger.info("=" * 60)
    logger.info("FNI ETL Pipeline — COMPLETE")
    logger.info("=" * 60)


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Modify PipelineConfig here for one-off runs.
    # For production, drive this via Airflow / Prefect / CLI args.
    pipeline_cfg = PipelineConfig(
        time_from=TIME_FROM,
        time_to=TIME_TO,
        skip_extract=False,
        skip_label=False,
        skip_score=False,
        skip_load=False,
    )
    run_pipeline(pipeline_cfg)