import os
import sys
import psycopg2
from psycopg2.extensions import connection as PgConnection
from dotenv import load_dotenv
from src.fni.core.logger import get_logger
from src.fni.core.exceptions import CustomException

load_dotenv()
logger = get_logger(__name__)


def get_db_connection() -> PgConnection:
    """
    Returns a live psycopg2 connection using environment variables.
    Caller is responsible for closing the connection.
    """
    try:
        conn: PgConnection = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "fni"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            connect_timeout=10,
        )
        logger.info("PostgreSQL connection established.")
        return conn
    except Exception as e:
        raise CustomException(f"Failed to connect to PostgreSQL: {e}", sys) from e