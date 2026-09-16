from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    database_url: str = "sqlite:///./loop_engine.db"
    max_rounds: int = 3
    early_stop_score: float = 9.0
    model_provider: str = "mock"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-pro-preview"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )


def get_settings() -> Settings:
    return Settings()
