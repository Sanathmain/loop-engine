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

    async def review(
        self,
        problem: str,
        solution: WriterOutput,
        previous_critique: CriticOutput | None = None,
        previous_score: float | None = None,
        previous_solution: WriterOutput | None = None,
        user_unsatisfied: bool = False,
    ) -> CriticOutput:
        parts = [
            f"Problem:\n{problem}",
            f"Writer solution:\n{solution.model_dump_json(indent=2)}",
        ]
        if previous_solution is not None:
            parts.append(
                "Previous Writer solution (compare for real change):\n"
                + previous_solution.model_dump_json(indent=2)
            )
        if previous_critique is not None:
            score = (
                previous_score
                if previous_score is not None
                else previous_critique.score
            )
            parts.append(
                "Previous critique (compare against this):\n"
                + previous_critique.model_dump_json(indent=2)
            )
            parts.append(
                f"Previous score: {score}\n"
                "Set verdict to improved, unchanged, or regressed. "
                "Fill resolved_points, regressions, and blocking_issues."
            )
        else:
            parts.append(
                "This is the first critique. Set verdict to unchanged, "
                "leave resolved_points and regressions empty, and put only "
                "must-fix problems in blocking_issues."
            )

        if user_unsatisfied:
            parts.append(
                "USER CONTINUATION MODE:\n"
                "- The user was not satisfied with the prior answer.\n"
                "- If this draft is mostly the same idea with cosmetic edits, "
                "set verdict=unchanged and do NOT raise the score.\n"
                "- Demand concrete new substance on open issues."
            )

        result = await self.generate(
            [{"role": "user", "content": "\n\n".join(parts)}],
            response_model=CriticOutput,
        )
        if not isinstance(result, CriticOutput):
            raise TypeError("Critic model did not return CriticOutput")
        return result
