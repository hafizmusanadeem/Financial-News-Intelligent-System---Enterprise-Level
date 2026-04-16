from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Financial News Intelligence"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # Database
    POSTGRES_USER: str = "None"
    POSTGRES_PASSWORD: str = "None"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "None"

    # NewsAPI
    NEWSAPI_KEY: str = "None"

    # Ingestion settings
    INGESTION_INTERVAL_MINUTES: int = 15
    MIN_ARTICLE_WORD_COUNT: int = 100
    IMPACT_SCORE_THRESHOLD: int = 70

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"

        
settings = Settings()