from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import logging
import time

from sqlalchemy.orm import Session

from app.agents.critic import CriticAgent
from app.agents.writer import WriterAgent
from app.database.models import SessionRecord
from app.schemas.critic import CriticOutput
from app.schemas.session import (
    HistoryItem,
    LoopEvent,
    LoopEventType,
    SessionRead,
    SessionStatus,
    SessionSummary,
)
from app.schemas.writer import WriterOutput

logger = logging.getLogger("loop_engine")

ABSOLUTE_MAX_ROUNDS = 20


class SessionNotFoundError(Exception):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionContinueError(Exception):
    """Raised when a session cannot be continued."""


class Orchestrator:
    """Runs the Writer → Critic loop for a stored session."""

    def __init__(
        self,
        db: Session,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
        max_rounds: int = 3,
        early_stop_score: float = 9.0,
        improvement_epsilon: float = 0.1,
    ) -> None:
        self.db = db
        self.writer = writer
        self.critic = critic
        self.max_rounds = max_rounds
        self.early_stop_score = early_stop_score
        self.improvement_epsilon = improvement_epsilon

    def get_session(self, session_id: str) -> SessionRecord:
        record = self.db.get(SessionRecord, session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        return record

    def list_sessions(self) -> list[SessionSummary]:
        rows = (
            self.db.query(SessionRecord)
            .order_by(SessionRecord.created_at.desc())
            .all()
        )
        return [record_to_summary(row) for row in rows]

    def create_session(
        self,
        problem: str,
        session_id: str,
        max_rounds: int | None = None,
    ) -> SessionRecord:
        record = SessionRecord(
            id=session_id,
            problem=problem,
            status=SessionStatus.RUNNING.value,
            current_round=0,
            max_rounds=max_rounds if max_rounds is not None else self.max_rounds,
            current_solution=None,
            history=[],
            final_answer=None,
            best_round=None,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        logger.info("Session created: %s", record.id)
        return record

    def prepare_continue(
        self,
        session_id: str,
        extra_rounds: int = 2,
    ) -> SessionRecord:
        """Raise max_rounds and reopen a finished/stopped session for more work."""
        record = self.get_session(session_id)
        history = record.history or []
        if not history and record.current_round < 1:
            raise SessionContinueError(
                "Session has no rounds to continue from. Start a debate first."
            )

        new_max = record.current_round + extra_rounds
        if new_max > ABSOLUTE_MAX_ROUNDS:
            new_max = ABSOLUTE_MAX_ROUNDS
        if new_max <= record.current_round:
            raise SessionContinueError(
                f"Session already reached the absolute limit of {ABSOLUTE_MAX_ROUNDS} rounds."
            )

        record.max_rounds = new_max
        record.status = SessionStatus.RUNNING.value
        record.final_answer = None
        record.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(record)
        logger.info(
            "Continuing session %s for %s more round(s) (max_rounds=%s)",
            session_id,
            extra_rounds,
            new_max,
        )
        return record

    async def run(self, session_id: str) -> SessionRead:
        async for event in self.iter_run(session_id):
            if event.type == LoopEventType.ERROR:
                raise RuntimeError((event.payload or {}).get("detail") or "Loop failed")
        return record_to_schema(self.get_session(session_id))

    async def iter_run(
        self,
        session_id: str,
        user_unsatisfied: bool = False,
    ) -> AsyncIterator[LoopEvent]:
        record = self.get_session(session_id)

        if record.status == SessionStatus.COMPLETED.value:
            yield LoopEvent(
                type=LoopEventType.COMPLETED,
                payload=record_to_schema(record).model_dump(mode="json"),
            )
            return

        if self.writer is None or self.critic is None:
            raise RuntimeError("Writer and Critic agents are required to run the loop")

        record.status = SessionStatus.RUNNING.value
        session_max_rounds = record.max_rounds or self.max_rounds
        self.db.commit()

        previous_solution: WriterOutput | None = None
        last_critique: CriticOutput | None = None
        history_models: list[HistoryItem] = [
            HistoryItem.model_validate(item) for item in (record.history or [])
        ]

        if history_models:
            last_item = history_models[-1]
            previous_solution = (
                last_item.revised_writer_response or last_item.writer_response
            )
            last_critique = last_item.critic_response

        best_score = -1.0
        best_round: int | None = record.best_round
        best_solution: WriterOutput | None = None
        if history_models:
            for item in history_models:
                if item.critic_response.score > best_score:
                    best_score = item.critic_response.score
                    best_round = item.round_number
                    best_solution = (
                        item.revised_writer_response or item.writer_response
                    )

        plateau_count = 0
        regression_count = 0
        # After an explicit Continue, force a fresh pass for the newly added rounds.
        force_fresh = user_unsatisfied and bool(history_models)
        continue_start_round: int | None = None

        try:
            start_round = record.current_round + 1
            if force_fresh:
                continue_start_round = start_round
                # User rejected the prior answer — do not keep crowning the old best.
                best_score = -1.0
                best_round = None
                best_solution = None
            for round_number in range(start_round, session_max_rounds + 1):
                logger.info("Round %s", round_number)
                yield LoopEvent(type=LoopEventType.ROUND, round_number=round_number)

                if previous_solution is None:
                    logger.info("Writer generating...")
                else:
                    logger.info(
                        "Writer revising%s...",
                        " (user unsatisfied)" if force_fresh else "",
                    )

                writer_started = time.perf_counter()
                writer_output = await self.writer.solve(
                    problem=record.problem,
                    previous_solution=previous_solution,
                    critique=last_critique,
                    history=history_models or None,
                    user_unsatisfied=force_fresh,
                )
                writer_stats = _call_stats(self.writer.model)
                writer_ms = int((time.perf_counter() - writer_started) * 1000)
                writer_ms = writer_stats.get("duration_ms") or writer_ms
                writer_tokens = writer_stats.get("tokens")

                record.current_solution = writer_output.model_dump(mode="json")
                record.updated_at = datetime.utcnow()
                self.db.commit()

                yield LoopEvent(
                    type=LoopEventType.WRITER,
                    round_number=round_number,
                    payload={
                        **writer_output.model_dump(mode="json"),
                        "duration_ms": writer_ms,
                        "tokens": writer_tokens,
                    },
                )

                logger.info("Critic reviewing...")
                critic_started = time.perf_counter()
                critic_output = await self.critic.review(
                    problem=record.problem,
                    solution=writer_output,
                    previous_critique=last_critique,
                    previous_score=(
                        last_critique.score if last_critique is not None else None
                    ),
                    previous_solution=previous_solution,
                    user_unsatisfied=force_fresh,
                )
                critic_stats = _call_stats(self.critic.model)
                critic_ms = int((time.perf_counter() - critic_started) * 1000)
                critic_ms = critic_stats.get("duration_ms") or critic_ms
                critic_tokens = critic_stats.get("tokens")
                logger.info("Critic score: %s", critic_output.score)

                duration_ms = (writer_ms or 0) + (critic_ms or 0)
                tokens = None
                if writer_tokens is not None or critic_tokens is not None:
                    tokens = (writer_tokens or 0) + (critic_tokens or 0)

                history_item = HistoryItem(
                    round_number=round_number,
                    writer_response=writer_output,
                    critic_response=critic_output,
                    revised_writer_response=writer_output if round_number > 1 else None,
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    tokens=tokens,
                )
                history_models.append(history_item)

                record.current_round = round_number
                record.history = [
                    item.model_dump(mode="json") for item in history_models
                ]
                record.updated_at = datetime.utcnow()

                if critic_output.score > best_score:
                    best_score = critic_output.score
                    best_round = round_number
                    best_solution = writer_output
                    record.best_round = best_round

                self.db.commit()

                yield LoopEvent(
                    type=LoopEventType.CRITIC,
                    round_number=round_number,
                    payload={
                        **critic_output.model_dump(mode="json"),
                        "duration_ms": critic_ms,
                        "tokens": critic_tokens,
                    },
                )

                previous_score = (
                    last_critique.score if last_critique is not None else None
                )
                previous_solution = writer_output
                last_critique = critic_output

                if previous_score is not None:
                    if critic_output.score + self.improvement_epsilon < previous_score:
                        regression_count += 1
                        plateau_count = 0
                    elif (
                        abs(critic_output.score - previous_score)
                        < self.improvement_epsilon
                    ):
                        plateau_count += 1
                    else:
                        plateau_count = 0
                        regression_count = 0

                if self._should_stop(
                    critic_output,
                    plateau_count=plateau_count,
                    regression_count=regression_count,
                    allow_score_stop=not force_fresh,
                    allow_plateau_stop=not force_fresh,
                ):
                    logger.info("Stopping early after round %s", round_number)
                    break

            if force_fresh and continue_start_round is not None:
                # Prefer the best among newly generated rounds; never fall back
                # to a pre-continue draft the user already rejected.
                new_items = [
                    item
                    for item in history_models
                    if item.round_number >= continue_start_round
                ]
                if new_items:
                    best_new = max(
                        new_items, key=lambda item: item.critic_response.score
                    )
                    best_round = best_new.round_number
                    best_solution = (
                        best_new.revised_writer_response or best_new.writer_response
                    )

            final = best_solution or previous_solution
            record.final_answer = final.model_dump(mode="json") if final else None
            record.best_round = best_round
            record.status = SessionStatus.COMPLETED.value
            record.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(record)

            logger.info("Loop completed.")
            logger.info("Final answer generated.")
            yield LoopEvent(
                type=LoopEventType.COMPLETED,
                payload=record_to_schema(record).model_dump(mode="json"),
            )
        except Exception as exc:
            record.status = SessionStatus.FAILED.value
            record.updated_at = datetime.utcnow()
            self.db.commit()
            logger.exception("Loop failed for session %s", session_id)
            yield LoopEvent(
                type=LoopEventType.ERROR,
                payload={"detail": str(exc)},
            )

    def _should_stop(
        self,
        critique: CriticOutput,
        *,
        plateau_count: int,
        regression_count: int,
        allow_score_stop: bool = True,
        allow_plateau_stop: bool = True,
    ) -> bool:
        if (
            allow_score_stop
            and critique.score >= self.early_stop_score
            and not critique.blocking_issues
        ):
            logger.info("Early stop: strong score with no blocking issues.")
            return True
        if allow_plateau_stop and plateau_count >= 2:
            logger.info("Early stop: score plateaued for two rounds.")
            return True
        if regression_count >= 2:
            logger.info("Early stop: two regressions detected.")
            return True
        return False


def _call_stats(model: object) -> dict:
    stats = getattr(model, "last_call_stats", None)
    return dict(stats) if isinstance(stats, dict) else {}


def _best_from_history(history: list[HistoryItem]) -> tuple[int | None, float | None]:
    if not history:
        return None, None
    best = max(history, key=lambda item: item.critic_response.score)
    return best.round_number, best.critic_response.score


def record_to_schema(record: SessionRecord) -> SessionRead:
    history = [HistoryItem.model_validate(item) for item in (record.history or [])]
    best_round = record.best_round
    if best_round is None and history:
        best_round, _ = _best_from_history(history)
    return SessionRead(
        id=record.id,
        problem=record.problem,
        status=SessionStatus(record.status),
        current_round=record.current_round,
        max_rounds=record.max_rounds,
        current_solution=(
            WriterOutput.model_validate(record.current_solution)
            if record.current_solution
            else None
        ),
        history=history,
        final_answer=(
            WriterOutput.model_validate(record.final_answer)
            if record.final_answer
            else None
        ),
        best_round=best_round,
        created_at=record.created_at,
    )


def record_to_summary(record: SessionRecord) -> SessionSummary:
    history = [HistoryItem.model_validate(item) for item in (record.history or [])]
    best_round, best_score = _best_from_history(history)
    if record.best_round is not None:
        best_round = record.best_round
        for item in history:
            if item.round_number == best_round:
                best_score = item.critic_response.score
                break
    return SessionSummary(
        id=record.id,
        problem=record.problem,
        status=SessionStatus(record.status),
        current_round=record.current_round,
        max_rounds=record.max_rounds,
        best_score=best_score,
        best_round=best_round,
        created_at=record.created_at,
    )
