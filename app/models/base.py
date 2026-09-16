from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class AIModel(ABC):
    """Provider-independent interface for any LLM backend."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        """Generate a completion from a list of chat messages.

        If ``response_model`` is provided, the result should be an instance of
        that Pydantic model. Otherwise a plain string is returned.
        """
