from __future__ import annotations

from app.agents.base import Agent
from app.models.base import AIModel
from app.prompts.writer_prompt import WRITER_SYSTEM_PROMPT
from app.schemas.critic import CriticOutput
from app.schemas.writer import WriterOutput


class WriterAgent(Agent):
    def __init__(self, model: AIModel, name: str = "Writer") -> None:
        super().__init__(
            name=name,
            role="writer",
            model=model,
            system_prompt=WRITER_SYSTEM_PROMPT,
        )

    async def solve(
        self,
        problem: str,
        previous_solution: WriterOutput | None = None,
        critique: CriticOutput | None = None,
    ) -> WriterOutput:
        parts = [f"Problem:\n{problem}"]
        if previous_solution is not None:
            parts.append(
                "Previous solution:\n"
                + previous_solution.model_dump_json(indent=2)
            )
        if critique is not None:
            parts.append("Latest critique:\n" + critique.model_dump_json(indent=2))

        result = await self.generate(
            [{"role": "user", "content": "\n\n".join(parts)}],
            response_model=WriterOutput,
        )
        if not isinstance(result, WriterOutput):
            raise TypeError("Writer model did not return WriterOutput")
        return result
