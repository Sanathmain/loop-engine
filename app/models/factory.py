from __future__ import annotations

from app.config import Settings
from app.models.base import AIModel
from app.models.mock_model import MockModel


def create_model(settings: Settings) -> AIModel:
    """Build an AIModel from settings. Add OpenAI/Anthropic branches here later."""
    provider = settings.model_provider.strip().lower()
    if provider == "mock":
        return MockModel()
    if provider == "gemini":
        from app.models.gemini_model import GeminiModel

        if not settings.gemini_api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required when MODEL_PROVIDER=gemini. "
                "Get a key from https://aistudio.google.com/apikey and put it in .env"
            )
        return GeminiModel(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            timeout_seconds=settings.gemini_timeout_seconds,
        )
    raise ValueError(
        f"Unknown MODEL_PROVIDER={settings.model_provider!r}. "
        "Use 'mock' or 'gemini' (openai/anthropic can be added later)."
    )
