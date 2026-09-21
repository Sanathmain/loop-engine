from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel, Field

from app.config import Settings
from app.models.factory import create_model
from app.models.gemini_model import GeminiModel, _split_messages
from app.models.mock_model import MockModel


class _Sample(BaseModel):
    answer: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(self.status_code, text=self.text, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse] | None = None) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses or [])

    async def post(self, url, params=None, json=None, headers=None):
        self.calls.append(
            {"url": url, "params": params, "json": json, "headers": headers}
        )
        if self._responses:
            return self._responses.pop(0)
        return _FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '{"answer": "ok", "confidence": 0.8}'}]}}
                ],
                "usageMetadata": {"totalTokenCount": 42},
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
async def test_gemini_generate_uses_header_not_query_key() -> None:
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

    assert result.answer == "ok"
    call = fake.calls[0]
    assert call["params"] is None
    assert call["headers"]["x-goog-api-key"] == "test-key"
    assert "gemini-2.5-flash:generateContent" in call["url"]
    assert call["json"]["systemInstruction"]["parts"][0]["text"] == "You are the Critic"
    assert model.last_call_stats["tokens"] == 42


@pytest.mark.asyncio
async def test_gemini_repairs_validation_error() -> None:
    fake = _FakeClient(
        responses=[
            _FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": '{"answer": "bad", "confidence": 95}'}]
                            }
                        }
                    ]
                }
            ),
            _FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": '{"answer": "fixed", "confidence": 0.9}'}]
                            }
                        }
                    ]
                }
            ),
        ]
    )
    model = GeminiModel(api_key="test-key", http_client=fake)
    result = await model.generate(
        [{"role": "user", "content": "hello"}],
        response_model=_Sample,
    )
    assert result.answer == "fixed"
    assert result.confidence == 0.9
    assert len(fake.calls) == 2
    repair_text = fake.calls[1]["json"]["contents"][-1]["parts"][0]["text"]
    assert "rejected by schema validation" in repair_text


@pytest.mark.asyncio
async def test_gemini_retries_on_429(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("app.models.gemini_model.asyncio.sleep", fake_sleep)

    fake = _FakeClient(
        responses=[
            _FakeResponse({"error": "rate"}, status_code=429, text="rate limited"),
            _FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": '{"answer": "ok", "confidence": 0.5}'}]
                            }
                        }
                    ]
                }
            ),
        ]
    )
    model = GeminiModel(api_key="test-key", http_client=fake, max_retries=2)
    result = await model.generate(
        [{"role": "user", "content": "hello"}],
        response_model=_Sample,
    )
    assert result.answer == "ok"
    assert sleeps == [1]
    assert len(fake.calls) == 2
