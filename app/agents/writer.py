from __future__ import annotations

from app.agents.base import Agent
from app.models.base import AIModel
from app.prompts.writer_prompt import WRITER_SYSTEM_PROMPT
from app.schemas.critic import CriticOutput
from app.schemas.session import HistoryItem
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
        history: list[HistoryItem] | None = None,
        user_unsatisfied: bool = False,
    ) -> WriterOutput:
        parts = [f"Problem:\n{problem}"]

        if history:
            parts.append(self._format_history(history))
            if user_unsatisfied:
                parts.append(
                    "USER NOT SATISFIED — continuation rules:\n"
                    "- The user rejected the current best answer and wants real progress.\n"
                    "- Do NOT restate, paraphrase, or lightly reword the previous solution.\n"
                    "- Keep only the parts the Critic explicitly praised; rebuild the rest.\n"
                    "- Address every open blocking_issue and recommended_change with concrete new steps.\n"
                    "- If the score plateaued, change strategy: different architecture, process, "
                    "or tradeoffs — not the same plan with more adjectives.\n"
                    "- In analysis, start with a short 'What is new in this draft:' bullet list."
                )
            else:
                parts.append(
                    "Revision rules:\n"
                    "- Keep everything prior Critic rounds praised.\n"
                    "- Change only criticized or incorrect parts.\n"
                    "- Do not shorten or drop good content unless asked.\n"
                    "- Use the strongest prior draft as the baseline.\n"
                    "- If your draft would look almost identical to the last one, you must add "
                    "concrete new fixes instead of rephrasing."
                )
        else:
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

    def _format_history(self, history: list[HistoryItem]) -> str:
        blocks: list[str] = ["Prior rounds (oldest first):"]
        for item in history:
            writer = item.revised_writer_response or item.writer_response
            critic = item.critic_response
            blocks.append(
                f"### Round {item.round_number}\n"
                f"Writer solution:\n{writer.solution}\n\n"
                f"Writer confidence: {writer.confidence}\n"
                f"Writer confidence_rationale: {writer.confidence_rationale}\n"
                f"Critic score: {critic.score}\n"
                f"Critic strengths: {critic.strengths}\n"
                f"Critic weaknesses: {critic.weaknesses}\n"
                f"Critic recommended_changes: {critic.recommended_changes}\n"
                f"Critic blocking_issues: {critic.blocking_issues}\n"
                f"Critic regressions: {critic.regressions}"
            )
        return "\n\n".join(blocks)
