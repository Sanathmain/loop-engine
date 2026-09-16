from __future__ import annotations

from app.agents.base import Agent
from app.models.base import AIModel
from app.prompts.critic_prompt import CRITIC_SYSTEM_PROMPT
from app.schemas.critic import CriticOutput
from app.schemas.writer import WriterOutput


class CriticAgent(Agent):
    def __init__(self, model: AIModel, name: str = "Critic") -> None:
        super().__init__(
            name=name,
            role="critic",
            model=model,
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )

    async def review(self, problem: str, solution: WriterOutput) -> CriticOutput:
        user_content = (
            f"Problem:\n{problem}\n\n"
            f"Writer solution:\n{solution.model_dump_json(indent=2)}"
        )
        result = await self.generate(
            [{"role": "user", "content": user_content}],
            response_model=CriticOutput,
        )
        if not isinstance(result, CriticOutput):
            raise TypeError("Critic model did not return CriticOutput")
        return result
