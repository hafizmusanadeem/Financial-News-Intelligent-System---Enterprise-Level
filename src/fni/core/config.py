from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Financial News Intelligence"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"

        
settings = Settings()