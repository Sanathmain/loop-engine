from __future__ import annotations

from typing import Any

import pytest

from app.agents.critic import CriticAgent
from app.agents.writer import WriterAgent
from app.models.base import AIModel
from app.models.mock_model import MockModel
from app.orchestrator.engine import Orchestrator, SessionNotFoundError
from app.schemas.critic import CriticOutput
from app.schemas.writer import WriterOutput
from tests.conftest import PROBLEM, make_session


def test_writer_confidence_percent_is_normalized() -> None:
    output = WriterOutput(
        analysis="a",
        solution="s",
        assumptions=[],
        risks=[],
        confidence=95,
    )
    assert output.confidence == 0.95


@pytest.mark.asyncio
async def test_run_completes_three_rounds(orchestrator: Orchestrator) -> None:
    session_id = make_session(orchestrator)
    result = await orchestrator.run(session_id)

    assert result.status.value == "completed"
    assert result.current_round == 3
    assert len(result.history) == 3
    assert result.final_answer is not None
    assert result.current_solution is not None
    assert result.final_answer.solution == result.history[-1].writer_response.solution

    scores = [item.critic_response.score for item in result.history]
    assert scores == [6.5, 8.0, 8.7]

    assert result.history[0].revised_writer_response is None
    assert result.history[1].revised_writer_response is not None
    assert result.history[2].revised_writer_response is not None


@pytest.mark.asyncio
async def test_run_is_idempotent_after_completion(orchestrator: Orchestrator) -> None:
    session_id = make_session(orchestrator)
    first = await orchestrator.run(session_id)
    second = await orchestrator.run(session_id)

    assert second.status.value == "completed"
    assert first.final_answer == second.final_answer
    assert len(second.history) == 3


@pytest.mark.asyncio
async def test_early_stop_when_score_high_and_no_weaknesses(db) -> None:
    model = MockModel(critic_scores=[9.5])
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=3,
        early_stop_score=9.0,
    )
    session_id = make_session(orchestrator)
    result = await orchestrator.run(session_id)

    assert result.status.value == "completed"
    assert result.current_round == 1
    assert len(result.history) == 1
    assert result.history[0].critic_response.score == 9.5
    assert result.history[0].critic_response.weaknesses == []


@pytest.mark.asyncio
async def test_no_early_stop_when_high_score_has_weaknesses(db) -> None:
    class StubModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0

        async def generate(
            self,
            messages: list[dict[str, Any]],
            response_model: type | None = None,
        ) -> WriterOutput | CriticOutput:
            system = messages[0]["content"] if messages else ""
            if "You are the Writer" in system or response_model is WriterOutput:
                self.writer_calls += 1
                return WriterOutput(
                    analysis="analysis",
                    solution=f"solution {self.writer_calls}",
                    assumptions=["a"],
                    risks=["r"],
                    confidence=0.5,
                )
            self.critic_calls += 1
            return CriticOutput(
                strengths=["s"],
                weaknesses=["still a major gap"],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=["fix it"],
                score=9.5,
            )

    model = StubModel()
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=3,
        early_stop_score=9.0,
    )
    session_id = make_session(orchestrator)
    result = await orchestrator.run(session_id)

    assert result.current_round == 3
    assert len(result.history) == 3


@pytest.mark.asyncio
async def test_failed_status_when_model_raises(db) -> None:
    class BoomModel(AIModel):
        async def generate(self, messages, response_model=None):
            raise RuntimeError("model unavailable")

    model = BoomModel()
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
    )
    session_id = make_session(orchestrator)

    with pytest.raises(RuntimeError, match="model unavailable"):
        await orchestrator.run(session_id)

    stored = orchestrator.get_session(session_id)
    assert stored.status == "failed"


def test_missing_session_raises(orchestrator: Orchestrator) -> None:
    with pytest.raises(SessionNotFoundError):
        orchestrator.get_session("missing-id")


@pytest.mark.asyncio
async def test_iter_run_yields_writer_then_critic(orchestrator: Orchestrator) -> None:
    session_id = make_session(orchestrator)
    types: list[str] = []
    async for event in orchestrator.iter_run(session_id):
        types.append(event.type.value)

    assert types[0] == "round"
    assert types[1] == "writer"
    assert types[2] == "critic"
    assert types.count("writer") == 3
    assert types.count("critic") == 3
    assert types[-1] == "completed"
