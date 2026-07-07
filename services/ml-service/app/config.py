"""Application settings loaded from environment / .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgres://postgres:postgres@localhost:5433/classroom"
    model_name: str = "yolo11m-pose.pt"
    device: str = "mps"


@lru_cache
def get_settings() -> Settings:
    return Settings()
