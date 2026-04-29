import sys
from src.fni.core.logger import setup_logger
from src.fni.core.logger import get_logger
from src.fni.core.exceptions import CustomException

setup_logger()
get_logger('INFO')

logger = get_logger(__name__)

try:
    logger.info("Logging Module Running")
    print("Function Running Smoothly")
    logger.debug("Success")
except:
    pass