from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.models.factory import create_model
from app.models.gemini_model import GeminiModel, _split_messages
from app.models.mock_model import MockModel


class _Sample(BaseModel):
    answer: str


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def post(self, url, params=None, json=None):
        self.calls.append({"url": url, "params": params, "json": json})
        return _FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '{"answer": "ok"}'}]}}
                ]
            }
        )


def test_factory_returns_mock() -> None:
    model = create_model(Settings(model_provider="mock"))
    assert isinstance(model, MockModel)


def test_factory_gemini_requires_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_model(Settings(model_provider="gemini", gemini_api_key=""))


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown MODEL_PROVIDER"):
        create_model(Settings(model_provider="openai"))


def test_split_messages_extracts_system_prompt() -> None:
    system, contents = _split_messages(
        [
            {"role": "system", "content": "You are the Writer"},
            {"role": "user", "content": "Problem: wait times"},
        ]
    )
    assert system == "You are the Writer"
    assert contents == [
        {"role": "user", "parts": [{"text": "Problem: wait times"}]}
    ]


@pytest.mark.asyncio
async def test_gemini_generate_posts_schema_and_parses_json() -> None:
    fake = _FakeClient()
    model = GeminiModel(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        http_client=fake,
    )

    result = await model.generate(
        [
            {"role": "system", "content": "You are the Critic"},
            {"role": "user", "content": "Review this"},
        ],
        response_model=_Sample,
    )

    assert result == _Sample(answer="ok")
    call = fake.calls[0]
    assert call["params"] == {"key": "test-key"}
    assert "gemini-2.5-flash:generateContent" in call["url"]
    assert call["json"]["systemInstruction"]["parts"][0]["text"] == "You are the Critic"
    assert call["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert call["json"]["generationConfig"]["responseSchema"]["type"] == "OBJECT"
