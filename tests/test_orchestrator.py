from __future__ import annotations

from typing import Any

import pytest

from app.agents.critic import CriticAgent
from app.agents.writer import WriterAgent
from app.models.base import AIModel
from app.models.mock_model import MockModel
from app.orchestrator.engine import Orchestrator, SessionNotFoundError
from app.schemas.critic import CriticOutput
from app.schemas.session import SessionRead
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
    assert result.best_round == 3

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
async def test_early_stop_when_score_high_and_no_blocking_issues(db) -> None:
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
    assert result.history[0].critic_response.blocking_issues == []


@pytest.mark.asyncio
async def test_no_early_stop_when_high_score_has_blocking_issues(db) -> None:
    class StubModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0
            self.last_call_stats = {"duration_ms": 1, "tokens": 5}

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
                blocking_issues=["must fix this first"],
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
async def test_best_round_is_selected_not_last(db) -> None:
    class StubModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0
            self.scores = [8.0, 4.0, 7.0]
            self.last_call_stats = {"duration_ms": 2, "tokens": 8}

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
                    solution=f"solution round {self.writer_calls}",
                    assumptions=["a"],
                    risks=["r"],
                    confidence=0.7,
                    confidence_rationale="stub",
                )
            score = self.scores[self.critic_calls]
            self.critic_calls += 1
            return CriticOutput(
                strengths=["s"],
                weaknesses=["w"],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=[],
                score=score,
                verdict="regressed" if score == 4.0 else "improved",
                blocking_issues=["open issue"],
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

    assert result.best_round == 1
    assert result.final_answer is not None
    assert result.final_answer.solution == "solution round 1"
    assert [item.critic_response.score for item in result.history] == [8.0, 4.0, 7.0]


@pytest.mark.asyncio
async def test_plateau_stops_after_two_flat_rounds(db) -> None:
    class StubModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0
            self.scores = [7.0, 7.05, 7.02, 7.01, 7.0]
            self.last_call_stats = {"duration_ms": 1, "tokens": 3}

        async def generate(
            self,
            messages: list[dict[str, Any]],
            response_model: type | None = None,
        ) -> WriterOutput | CriticOutput:
            system = messages[0]["content"] if messages else ""
            if "You are the Writer" in system or response_model is WriterOutput:
                self.writer_calls += 1
                return WriterOutput(
                    analysis="a",
                    solution=f"s{self.writer_calls}",
                    assumptions=[],
                    risks=[],
                    confidence=0.6,
                )
            score = self.scores[self.critic_calls]
            self.critic_calls += 1
            return CriticOutput(
                strengths=[],
                weaknesses=["w"],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=[],
                score=score,
                blocking_issues=["still open"],
            )

    model = StubModel()
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=5,
        early_stop_score=9.0,
        improvement_epsilon=0.1,
    )
    session_id = make_session(orchestrator)
    result = await orchestrator.run(session_id)

    assert result.current_round == 3
    assert len(result.history) == 3
    assert model.writer_calls == 3


@pytest.mark.asyncio
async def test_writer_prompt_includes_prior_rounds(db) -> None:
    captured: list[str] = []

    class CaptureModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0
            self.last_call_stats = {"duration_ms": 1, "tokens": 1}

        async def generate(
            self,
            messages: list[dict[str, Any]],
            response_model: type | None = None,
        ) -> WriterOutput | CriticOutput:
            system = messages[0]["content"] if messages else ""
            user = messages[-1]["content"] if messages else ""
            if "You are the Writer" in system or response_model is WriterOutput:
                self.writer_calls += 1
                captured.append(user)
                return WriterOutput(
                    analysis="a",
                    solution=f"writer {self.writer_calls}",
                    assumptions=[],
                    risks=[],
                    confidence=0.5,
                )
            self.critic_calls += 1
            return CriticOutput(
                strengths=["ok"],
                weaknesses=["gap"],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=["fix"],
                score=6.0 + self.critic_calls,
                blocking_issues=["gap"],
            )

    model = CaptureModel()
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=2,
    )
    session_id = make_session(orchestrator)
    await orchestrator.run(session_id)

    assert len(captured) == 2
    assert "Prior rounds" in captured[1]
    assert "Critic score" in captured[1]
    assert "Revision rules" in captured[1]


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
async def test_prepare_continue_raises_max_rounds(db) -> None:
    model = MockModel(critic_scores=[6.5, 7.0, 7.5, 8.0, 8.2])
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=2,
    )
    session_id = make_session(orchestrator)
    await orchestrator.run(session_id)
    record = orchestrator.prepare_continue(session_id, extra_rounds=3)
    assert record.max_rounds == 5
    assert record.status == "running"
    assert record.final_answer is None

    result = await orchestrator.run(session_id)
    assert result.current_round == 5
    assert len(result.history) == 5


@pytest.mark.asyncio
async def test_writer_prompt_includes_unsatisfied_rules(db) -> None:
    captured: list[str] = []

    class CaptureModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0
            self.last_call_stats = {"duration_ms": 1, "tokens": 1}

        async def generate(
            self,
            messages: list[dict[str, Any]],
            response_model: type | None = None,
        ) -> WriterOutput | CriticOutput:
            system = messages[0]["content"] if messages else ""
            user = messages[-1]["content"] if messages else ""
            if "You are the Writer" in system or response_model is WriterOutput:
                self.writer_calls += 1
                captured.append(user)
                return WriterOutput(
                    analysis="a",
                    solution=f"writer {self.writer_calls}",
                    assumptions=[],
                    risks=[],
                    confidence=0.5,
                )
            self.critic_calls += 1
            return CriticOutput(
                strengths=["ok"],
                weaknesses=["gap"],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=["fix"],
                score=6.0,
                blocking_issues=["gap"],
            )

    model = CaptureModel()
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=1,
    )
    session_id = make_session(orchestrator)
    await orchestrator.run(session_id)
    orchestrator.prepare_continue(session_id, extra_rounds=1)
    captured.clear()
    async for _ in orchestrator.iter_run(session_id, user_unsatisfied=True):
        pass
    assert captured
    assert "USER NOT SATISFIED" in captured[0]
    assert "Do NOT restate" in captured[0]


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


@pytest.mark.asyncio
async def test_continue_picks_best_among_new_rounds_only(db) -> None:
    """Continue must not reuse a pre-continue final answer the user rejected."""

    class ScriptedModel(AIModel):
        def __init__(self) -> None:
            self.writer_calls = 0
            self.critic_calls = 0
            self.last_call_stats = {"duration_ms": 1, "tokens": 1}

        async def generate(
            self,
            messages: list[dict[str, Any]],
            response_model: type | None = None,
        ) -> WriterOutput | CriticOutput:
            system = messages[0]["content"] if messages else ""
            if "You are the Writer" in system or response_model is WriterOutput:
                self.writer_calls += 1
                return WriterOutput(
                    analysis="a",
                    solution=f"draft-{self.writer_calls}",
                    assumptions=[],
                    risks=[],
                    confidence=0.5,
                )
            self.critic_calls += 1
            # First round scores very high; continue rounds score lower but must win.
            score = 9.5 if self.critic_calls == 1 else 6.0 + self.critic_calls * 0.1
            return CriticOutput(
                strengths=["ok"],
                weaknesses=["gap"],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=["fix"],
                score=score,
                blocking_issues=[] if self.critic_calls == 1 else ["gap"],
            )

    model = ScriptedModel()
    orchestrator = Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=1,
        early_stop_score=9.0,
    )
    session_id = make_session(orchestrator)
    first = await orchestrator.run(session_id)
    assert first.final_answer is not None
    assert first.final_answer.solution == "draft-1"
    assert first.best_round == 1

    orchestrator.prepare_continue(session_id, extra_rounds=2)
    last: SessionRead | None = None
    async for event in orchestrator.iter_run(session_id, user_unsatisfied=True):
        if event.type.value == "completed":
            last = SessionRead.model_validate(event.payload)
    assert last is not None
    assert last.final_answer is not None
    assert last.final_answer.solution != "draft-1"
    assert last.best_round is not None and last.best_round > 1
