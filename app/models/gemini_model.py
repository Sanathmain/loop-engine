from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.models.base import AIModel

logger = logging.getLogger("loop_engine.gemini")

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_JSON_TO_GEMINI_TYPE = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "object": "OBJECT",
    "array": "ARRAY",
    "null": "NULL",
}

_RETRY_STATUS = {429, 500, 503}


class GeminiModel(AIModel):
    """Gemini adapter via the public REST API (no extra native deps)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-pro-preview",
        http_client: Any | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._http_client = http_client
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.last_call_stats: dict[str, Any] = {}

    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        system_instruction, contents = _split_messages(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        generation_config: dict[str, Any] = {
            # Slightly higher temperature reduces near-identical Continue drafts.
            "temperature": 0.9,
        }
        if response_model is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _pydantic_to_gemini_schema(
                response_model
            )
        payload["generationConfig"] = generation_config

        started = time.perf_counter()
        data = await self._post_with_retries(payload)
        text = _extract_text(data)
        usage = data.get("usageMetadata") or {}
        self.last_call_stats = {
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "tokens": usage.get("totalTokenCount"),
            "prompt_tokens": usage.get("promptTokenCount"),
            "response_tokens": usage.get("candidatesTokenCount"),
        }

        if response_model is None:
            return text
        if not text.strip():
            raise ValueError("Gemini returned an empty response")

        try:
            return response_model.model_validate_json(_strip_json_fences(text))
        except ValidationError as exc:
            repaired = await self._repair_validation(
                payload=payload,
                response_model=response_model,
                bad_text=text,
                error=exc,
            )
            return repaired

    async def _repair_validation(
        self,
        *,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        bad_text: str,
        error: ValidationError,
    ) -> BaseModel:
        repair_payload = dict(payload)
        contents = list(repair_payload.get("contents") or [])
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Your previous JSON was rejected by schema validation.\n"
                            f"Validation error:\n{error}\n\n"
                            f"Invalid JSON:\n{bad_text}\n\n"
                            "Return corrected JSON that fully matches the schema. "
                            "confidence must be between 0.0 and 1.0, not a percentage."
                        )
                    }
                ],
            }
        )
        repair_payload["contents"] = contents
        data = await self._post_with_retries(repair_payload)
        text = _extract_text(data)
        if not text.strip():
            raise ValueError("Gemini returned an empty repair response") from error
        return response_model.model_validate_json(_strip_json_fences(text))

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._post(payload)
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                retryable = any(str(code) in message for code in _RETRY_STATUS)
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = 2**attempt
                logger.warning(
                    "Gemini retryable error (attempt %s/%s); sleeping %ss",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = GEMINI_GENERATE_URL.format(model=self.model_name)
        headers = {"x-goog-api-key": self.api_key}
        client = self._http_client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                raise RuntimeError(
                    f"Gemini API error ({exc.response.status_code}): {detail}"
                ) from exc
            data = response.json()
        finally:
            if owns_client:
                await client.aclose()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Gemini API error: {data['error']}")
        return data


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role") or "user"
        text = str(message.get("content") or "")
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        gemini_role = "model" if role in {"assistant", "model"} else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if not contents:
        raise ValueError("GeminiModel.generate requires at least one non-system message")
    system_instruction = "\n\n".join(system_parts) or None
    return system_instruction, contents


def _pydantic_to_gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    return _to_gemini_schema(model.model_json_schema())


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    json_type = schema.get("type")
    if json_type:
        converted["type"] = _JSON_TO_GEMINI_TYPE.get(str(json_type), str(json_type).upper())
    if "description" in schema:
        converted["description"] = schema["description"]
    if "enum" in schema:
        converted["enum"] = schema["enum"]
    if "properties" in schema:
        converted["properties"] = {
            name: _to_gemini_schema(prop) for name, prop in schema["properties"].items()
        }
    if "required" in schema:
        converted["required"] = schema["required"]
    if "items" in schema and isinstance(schema["items"], dict):
        converted["items"] = _to_gemini_schema(schema["items"])
    # Gemini structured output rejects additionalProperties in some schemas.
    if "$defs" in schema:
        # Inline defs are already expanded by pydantic for top-level models we use.
        pass
    return converted


def _extract_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [str(part.get("text") or "") for part in parts]
    return "".join(texts)


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return stripped.strip()
