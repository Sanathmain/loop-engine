from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from app.models.base import AIModel

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


class GeminiModel(AIModel):
    """Gemini adapter via the public REST API (no extra native deps)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-pro-preview",
        http_client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._http_client = http_client

    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        system_instruction, contents = _split_messages(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if response_model is not None:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": _pydantic_to_gemini_schema(response_model),
            }

        data = await self._post(payload)
        text = _extract_text(data)
        if response_model is None:
            return text
        if not text.strip():
            raise ValueError("Gemini returned an empty response")
        return response_model.model_validate_json(_strip_json_fences(text))

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = GEMINI_GENERATE_URL.format(model=self.model_name)
        params = {"key": self.api_key}
        client = self._http_client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=60.0)
        try:
            response = await client.post(url, params=params, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                raise RuntimeError(f"Gemini API error: {detail}") from exc
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
