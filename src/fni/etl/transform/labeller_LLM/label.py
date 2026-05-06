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

# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = Path(r'C:\Users\DELL\Documents\MLOPS\Finance News Intelligent System\Financial-News-Intelligent-System---Enterprise-Level\artifacts\interim\news_data.csv')
OUTPUT_PATH = Path(r'C:\Users\DELL\Documents\MLOPS\Finance News Intelligent System\Financial-News-Intelligent-System---Enterprise-Level\artifacts\labelled\news_data_labeled.csv')
REQUEST_DELAY = 3
# ============================================================

api_key = os.getenv("OpenRouter_API")
if not api_key:
    raise CustomException("OPENROUTER_API_KEY is not set in your .env file.", sys)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

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


def get_the_event_label(title: str, summary: str) -> str:
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
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
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logger.warning(f"Rate limited on {model} (attempt {attempt+1}) — waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"Model {model} failed: {e} — trying next model")
                    break  # non-429 error, move to next model

    raise CustomException(f"All models exhausted for title '{title[:60]}'", sys)


# ── Load input ────────────────────────────────────────────────────────────

try:
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded input dataset — shape: {df.shape}")
except Exception as e:
    raise CustomException(f"Failed to load input CSV: {e}", sys)


# ── Resume logic ──────────────────────────────────────────────────────────

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

already_labeled_count = 0

if OUTPUT_PATH.exists():
    try:
        df_done = pd.read_csv(OUTPUT_PATH)
        already_labeled_count = len(df_done)
        logger.info(f"Resuming — found {already_labeled_count} already-labeled rows in output file.")
    except Exception as e:
        raise CustomException(f"Failed to read existing output CSV for resume: {e}", sys)
else:
    logger.info("No existing output file found — starting fresh.")

df_remaining = df.iloc[already_labeled_count:].reset_index(drop=True)

if df_remaining.empty:
    logger.info("All rows are already labeled. Nothing to do.")
    sys.exit(0)

logger.info(f"Rows to label this run: {len(df_remaining)}")


# ── Label & append row-by-row ─────────────────────────────────────────────

for i, row in enumerate(df_remaining.itertuples(), 1):
    global_index = already_labeled_count + i  # for logging

    try:
        logger.info(f"[{global_index}/{len(df)}] Labeling: {str(row.title)[:50]}...")
        label = get_the_event_label(str(row.title), str(row.summary))
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(f"Unexpected error at row {global_index}: {e}", sys)

    row_dict = row._asdict()
    row_dict.pop("Index", None)
    row_dict["event_label"] = label

    row_df = pd.DataFrame([row_dict])

    try:
        write_header = not OUTPUT_PATH.exists()
        row_df.to_csv(OUTPUT_PATH, mode="a", header=write_header, index=False)
    except Exception as e:
        raise CustomException(f"Failed to write row {global_index} to output CSV: {e}", sys)

    time.sleep(REQUEST_DELAY)


# ── Final summary ─────────────────────────────────────────────────────────

try:
    df_final = pd.read_csv(OUTPUT_PATH)
    logger.info(f"Run complete — total labeled rows: {len(df_final)} | path: {OUTPUT_PATH}")
    logger.info(f"Label distribution:\n{df_final['event_label'].value_counts().to_string()}")
except Exception as e:
    raise CustomException(f"Failed to read final output for summary: {e}", sys)