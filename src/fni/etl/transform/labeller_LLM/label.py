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

VALID_LABELS = {
    "Earnings", "Merger/Acquisition", "Product Launch",
    "Regulatory/Legal", "Market Movement", "Leadership Change",
    "Economic Data", "Fed Monetary Policy", "Other"
}

PROMPT_TEMPLATE = """You are a financial news classifier.
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

MODELS = [
    "baidu/cobuddy:free",
    "openrouter/owl-alpha",
    "google/gemma-4-26b-a4b-it:free",
    "baidu/qianfan-ocr-fast:free",
    "minimax/minimax-m2.5:free"
]

input_path = r'C:\Users\DELL\Documents\MLOPS\Finance News Intelligent System\Financial-News-Intelligent-System---Enterprise-Level\artifacts\interim\news_data.csv'

output_path = r'C:\Users\DELL\Documents\MLOPS\Finance News Intelligent System\Financial-News-Intelligent-System---Enterprise-Level\artifacts\labelled\news_data_labeled.csv'


class EventLabeller:

    def __init__(
        self,
        input_path: str,
        output_path: str,
        request_delay: int = 3
    ):
        self.input_path    = Path(input_path)
        self.output_path   = Path(output_path)
        self.request_delay = request_delay

        api_key = os.getenv("OpenRouter_API")
        if not api_key:
            raise CustomException("OPENROUTER_API_KEY is not set in your .env file.", sys)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_the_event_label(self, title: str, summary: str) -> str:
        for model in MODELS:
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(
                            title=title, summary=summary
                        )}]
                    )
                    label = response.choices[0].message.content.strip()
                    if label not in VALID_LABELS:
                        logger.warning(f"Unexpected label '{label}' from {model} — marking as Other")
                        return "Other"
                    return label

                except Exception as e:
                    if "429" in str(e):
                        wait = 2 ** attempt * 5
                        logger.warning(f"Rate limited on {model} (attempt {attempt+1}) — waiting {wait}s")
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning(f"Model {model} failed: {e} — trying next model")
                        break

        raise CustomException(f"All models exhausted for title '{title[:60]}'", sys)

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded input dataset — shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(f"Failed to load input CSV: {e}", sys)

    def _get_resume_index(self) -> int:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.exists():
            try:
                df_done = pd.read_csv(self.output_path)
                already_labeled_count = len(df_done)
                logger.info(f"Resuming — found {already_labeled_count} already-labeled rows in output file.")
                return already_labeled_count
            except Exception as e:
                raise CustomException(f"Failed to read existing output CSV for resume: {e}", sys)
        else:
            logger.info("No existing output file found — starting fresh.")
            return 0

    def _label(self, df: pd.DataFrame, already_labeled_count: int):
        df_remaining = df.iloc[already_labeled_count:].reset_index(drop=True)

        if df_remaining.empty:
            logger.info("All rows are already labeled. Nothing to do.")
            sys.exit(0)

        logger.info(f"Rows to label this run: {len(df_remaining)}")

        for i, row in enumerate(df_remaining.itertuples(), 1):
            global_index = already_labeled_count + i

            try:
                logger.info(f"[{global_index}/{len(df)}] Labeling: {str(row.title)[:50]}...")
                label = self._get_the_event_label(str(row.title), str(row.summary))
            except CustomException:
                raise
            except Exception as e:
                raise CustomException(f"Unexpected error at row {global_index}: {e}", sys)

            row_dict = row._asdict()
            row_dict.pop("Index", None)
            row_dict["event_label"] = label

            row_df = pd.DataFrame([row_dict])

            try:
                write_header = not self.output_path.exists()
                row_df.to_csv(self.output_path, mode="a", header=write_header, index=False)
            except Exception as e:
                raise CustomException(f"Failed to write row {global_index} to output CSV: {e}", sys)

            time.sleep(self.request_delay)

    def _summarize(self):
        try:
            df_final = pd.read_csv(self.output_path)
            logger.info(f"Run complete — total labeled rows: {len(df_final)} | path: {self.output_path}")
            logger.info(f"Label distribution:\n{df_final['event_label'].value_counts().to_string()}")
        except Exception as e:
            raise CustomException(f"Failed to read final output for summary: {e}", sys)

    # ── Public entry point ────────────────────────────────────────────────

    def run(self):
        df                    = self._load()
        already_labeled_count = self._get_resume_index()
        self._label(df, already_labeled_count)
        self._summarize()


if __name__ == "__main__":
    labeller = EventLabeller(
        input_path=input_path,
        output_path=output_path,
    )
    labeller.run()