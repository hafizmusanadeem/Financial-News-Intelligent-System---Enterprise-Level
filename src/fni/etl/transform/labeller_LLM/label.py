"""
src/fni/etl/transform/labeller_LLM/label.py

Classifies each news item into one of the predefined financial event labels
using free OpenRouter LLM models with a model-waterfall fallback strategy.

Rate-limit behaviour:
    - 429 on a model  → jump to next model immediately (no wait)
    - 429 on ALL models in the waterfall → wait, then retry the full waterfall
    - Max full-waterfall retries: MAX_WATERFALL_RETRIES (default 3)

Resumable: tracks progress by counting rows already written to the output CSV.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.fni.core.logger import setup_logger, get_logger
from src.fni.core.exceptions import CustomException

load_dotenv()
setup_logger()
logger = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

VALID_LABELS: frozenset[str] = frozenset({
    "Earnings", "Merger/Acquisition", "Product Launch",
    "Regulatory/Legal", "Market Movement", "Leadership Change",
    "Economic Data", "Fed Monetary Policy", "Other",
})

PROMPT_TEMPLATE: str = """You are a financial news classifier.
Based on the title and summary below, assign ONE event label from this list:
- Earnings
- Merger/Acquisition
- Product Launch
- Regulatory/Legal
- Market Movement
- Leadership Change
- Economic Data
- Fed Monetary Policy
- Other

Title: {title}
Summary: {summary}

Respond with ONLY the label. No explanation, no punctuation."""

# Waterfall order — exhausted left-to-right before any wait is introduced.
MODELS: list[str] = [
    "baidu/cobuddy:free",
    "openrouter/owl-alpha",
    "google/gemma-4-26b-a4b-it:free",
    "baidu/qianfan-ocr-fast:free",
    "minimax/minimax-m2.5:free",
]

# How many times to retry the FULL waterfall after every model is rate-limited.
MAX_WATERFALL_RETRIES: int = 3

# Wait (seconds) between full-waterfall retry attempts.
# Only triggered when every model in MODELS returned 429 in the same pass.
WATERFALL_RETRY_WAIT: int = 30


# ── Labeller ──────────────────────────────────────────────────────────────────

class EventLabeller:

    def __init__(
        self,
        input_path: str,
        output_path: str,
        request_delay: int = 3,
    ) -> None:
        self.input_path: Path = Path(input_path)
        self.output_path: Path = Path(output_path)
        self.request_delay: int = request_delay

        api_key = os.getenv("OpenRouter_API")
        if not api_key:
            raise CustomException(
                "OpenRouter_API is not set in your .env file.", sys
            )

        self.client: OpenAI = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _try_model(self, model: str, title: str, summary: str) -> str | None:
        """
        Attempt a single model call.

        Returns:
            str   — valid label on success
            None  — model was rate-limited (caller should try next model immediately)

        Raises:
            CustomException — non-rate-limit failure (broken model / network issue)
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(title=title, summary=summary),
                }],
            )
            raw: str = (response.choices[0].message.content or "").strip()
            if raw not in VALID_LABELS:
                logger.warning(
                    f"Unexpected label '{raw}' from {model} — marking as Other"
                )
                return "Other"
            return raw

        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limited on {model} — jumping to next model")
                return None   # caller hops immediately, no sleep here
            logger.warning(f"Model {model} failed with non-429 error: {e}")
            raise CustomException(
                f"Model {model} failed (non-rate-limit): {e}", sys
            ) from e

    def _get_the_event_label(self, title: str, summary: str) -> str:
        """
        Waterfall strategy:

        For each attempt in 1..MAX_WATERFALL_RETRIES:
            Iterate through every model in MODELS:
                success  → return label immediately
                429      → continue to next model with zero wait
                error    → continue to next model

            After the full model list is exhausted in one pass:
                - If every model was rate-limited → wait WATERFALL_RETRY_WAIT, retry
                - If at least one had a non-429 error but none succeeded → raise

        Raises CustomException if all waterfall attempts fail.
        """
        for attempt in range(1, MAX_WATERFALL_RETRIES + 1):
            all_rate_limited: bool = True

            for model in MODELS:
                try:
                    result = self._try_model(model, title, summary)
                except CustomException:
                    # Non-429 error on this model — don't block the waterfall
                    all_rate_limited = False
                    continue

                if result is not None:
                    return result
                # result is None → 429, keep all_rate_limited=True and try next model

            # ── Every model in this pass was tried ────────────────────────────
            if all_rate_limited:
                if attempt < MAX_WATERFALL_RETRIES:
                    logger.warning(
                        f"All {len(MODELS)} models rate-limited "
                        f"(waterfall attempt {attempt}/{MAX_WATERFALL_RETRIES}). "
                        f"Waiting {WATERFALL_RETRY_WAIT}s before retrying ..."
                    )
                    time.sleep(WATERFALL_RETRY_WAIT)
                else:
                    raise CustomException(
                        f"All models rate-limited across {MAX_WATERFALL_RETRIES} "
                        f"full-waterfall attempts for title '{title[:60]}'.",
                        sys,
                    )
            else:
                # Mix of non-429 errors — no point retrying with a wait
                raise CustomException(
                    f"All models failed (non-rate-limit errors) "
                    f"for title '{title[:60]}'.",
                    sys,
                )

        # Unreachable — satisfies strict type checker
        raise CustomException(  # pragma: no cover
            f"Waterfall exhausted for title '{title[:60]}'.", sys
        )

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded input dataset — shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(f"Failed to load input CSV: {e}", sys) from e

    def _get_resume_index(self) -> int:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            try:
                df_done = pd.read_csv(self.output_path)
                count = len(df_done)
                logger.info(f"Resuming — found {count} already-labeled rows.")
                return count
            except Exception as e:
                raise CustomException(
                    f"Failed to read existing output CSV for resume: {e}", sys
                ) from e
        logger.info("No existing output file — starting fresh.")
        return 0

    def _label(self, df: pd.DataFrame, already_labeled_count: int) -> None:
        df_remaining = df.iloc[already_labeled_count:].reset_index(drop=True)
        total = len(df)

        if df_remaining.empty:
            logger.info("All rows are already labeled. Nothing to do.")
            return

        logger.info(f"Rows to label this run: {len(df_remaining)}")

        for i, row in enumerate(df_remaining.itertuples(), 1):
            global_index = already_labeled_count + i

            try:
                logger.info(
                    f"[{global_index}/{total}] Labeling: {str(row.title)[:50]}..."
                )
                label = self._get_the_event_label(str(row.title), str(row.summary))
            except CustomException:
                raise
            except Exception as e:
                raise CustomException(
                    f"Unexpected error at row {global_index}: {e}", sys
                ) from e

            row_dict: dict[str, object] = row._asdict()
            row_dict.pop("Index", None)
            row_dict["event_label"] = label
            row_df = pd.DataFrame([row_dict])

            try:
                write_header = not self.output_path.exists()
                row_df.to_csv(
                    self.output_path, mode="a", header=write_header, index=False
                )
            except Exception as e:
                raise CustomException(
                    f"Failed to write row {global_index} to output CSV: {e}", sys
                ) from e

            time.sleep(self.request_delay)

    def _summarize(self) -> None:
        try:
            df_final = pd.read_csv(self.output_path)
            logger.info(
                f"Run complete — total labeled rows: {len(df_final)} "
                f"| path: {self.output_path}"
            )
            logger.info(
                f"Label distribution:\n"
                f"{df_final['event_label'].value_counts().to_string()}"
            )
        except Exception as e:
            raise CustomException(
                f"Failed to read final output for summary: {e}", sys
            ) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        df = self._load()
        already_labeled_count = self._get_resume_index()
        self._label(df, already_labeled_count)
        self._summarize()