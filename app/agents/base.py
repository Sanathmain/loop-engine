from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.models.base import AIModel


class Agent:
    """Thin wrapper that binds a role, system prompt, and model."""

    def __init__(
        self,
        name: str,
        role: str,
        model: AIModel,
        system_prompt: str,
    ) -> None:
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt

    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        return await self.model.generate(messages, response_model=response_model)
