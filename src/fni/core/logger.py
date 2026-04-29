import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime



BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# CONFIGURATION
# ==========================================================
LOG_LEVEL = logging.INFO
MAX_BYTES = 5 * 1024 * 1024      # 5 MB
BACKUP_COUNT = 3                # Keep last 3 rotated files


# ==========================================================
# FORMATTERS
# ==========================================================
FILE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

CONSOLE_FORMAT = logging.Formatter(
    "%(levelname)-8s | %(name)s | %(message)s"
)


# ==========================================================
# OPTIONAL COLORED TERMINAL OUTPUT
# ==========================================================
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m"    # Magenta
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


# ==========================================================
# MAIN LOGGER SETUP
# To Call ONLY once in main.py / app.py
# ==========================================================
def setup_logger():
    root_logger = logging.getLogger()

    # Remove duplicate handlers on rerun / notebook / restart
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(LOG_LEVEL)

    # Create unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}.log"

    # ------------------------------------------------------
    # File Handler with Rotation
    # ------------------------------------------------------
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )

    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(FILE_FORMAT)

    # ------------------------------------------------------
    # Console Handler
    # ------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    color_formatter = ColorFormatter(
        "%(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler.setFormatter(color_formatter)

    # ------------------------------------------------------
    # Attach handlers
    # ------------------------------------------------------
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info("Logger initialized successfully")


# ==========================================================
# MODULE LOGGER HELPER
# Use in modules:
#
# from src.logger import get_logger
# logger = get_logger(__name__)
# ==========================================================
def get_logger(name: str):
    return logging.getLogger(name)