from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session

from app.agents.critic import CriticAgent
from app.agents.writer import WriterAgent
from app.database.models import SessionRecord
from app.schemas.critic import CriticOutput
from app.schemas.session import HistoryItem, LoopEvent, LoopEventType, SessionRead, SessionStatus
from app.schemas.writer import WriterOutput

logger = logging.getLogger("loop_engine")


class SessionNotFoundError(Exception):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class Orchestrator:
    """Runs the Writer → Critic loop for a stored session."""

    def __init__(
        self,
        db: Session,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
        max_rounds: int = 3,
        early_stop_score: float = 9.0,
    ) -> None:
        self.db = db
        self.writer = writer
        self.critic = critic
        self.max_rounds = max_rounds
        self.early_stop_score = early_stop_score

    def get_session(self, session_id: str) -> SessionRecord:
        record = self.db.get(SessionRecord, session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        return record

    def create_session(self, problem: str, session_id: str) -> SessionRecord:
        record = SessionRecord(
            id=session_id,
            problem=problem,
            status=SessionStatus.RUNNING.value,
            current_round=0,
            max_rounds=self.max_rounds,
            current_solution=None,
            history=[],
            final_answer=None,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        logger.info("Session created: %s", record.id)
        return record

    async def run(self, session_id: str) -> SessionRead:
        async for event in self.iter_run(session_id):
            if event.type == LoopEventType.ERROR:
                raise RuntimeError((event.payload or {}).get("detail") or "Loop failed")
        return record_to_schema(self.get_session(session_id))

    async def iter_run(self, session_id: str) -> AsyncIterator[LoopEvent]:
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
        record.max_rounds = self.max_rounds
        self.db.commit()

        previous_solution: WriterOutput | None = None
        last_critique: CriticOutput | None = None
        history: list[dict] = list(record.history or [])

        if history:
            last_item = HistoryItem.model_validate(history[-1])
            previous_solution = (
                last_item.revised_writer_response or last_item.writer_response
            )
            last_critique = last_item.critic_response

        try:
            start_round = record.current_round + 1
            for round_number in range(start_round, self.max_rounds + 1):
                logger.info("Round %s", round_number)
                yield LoopEvent(type=LoopEventType.ROUND, round_number=round_number)

                if previous_solution is None:
                    logger.info("Writer generating...")
                else:
                    logger.info("Writer revising...")

                writer_output = await self.writer.solve(
                    problem=record.problem,
                    previous_solution=previous_solution,
                    critique=last_critique,
                )
                record.current_solution = writer_output.model_dump(mode="json")
                record.updated_at = datetime.utcnow()
                self.db.commit()

                yield LoopEvent(
                    type=LoopEventType.WRITER,
                    round_number=round_number,
                    payload=writer_output.model_dump(mode="json"),
                )

                logger.info("Critic reviewing...")
                critic_output = await self.critic.review(
                    problem=record.problem,
                    solution=writer_output,
                )
                logger.info("Critic score: %s", critic_output.score)

                history_item = HistoryItem(
                    round_number=round_number,
                    writer_response=writer_output,
                    critic_response=critic_output,
                    revised_writer_response=writer_output if round_number > 1 else None,
                    timestamp=datetime.now(timezone.utc),
                )
                history.append(history_item.model_dump(mode="json"))

                record.current_round = round_number
                record.history = list(history)
                record.updated_at = datetime.utcnow()
                self.db.commit()

                yield LoopEvent(
                    type=LoopEventType.CRITIC,
                    round_number=round_number,
                    payload=critic_output.model_dump(mode="json"),
                )

                previous_solution = writer_output
                last_critique = critic_output

                if self._should_stop_early(critic_output):
                    logger.info("Early stop: critic score is strong enough.")
                    break

            record.final_answer = (
                previous_solution.model_dump(mode="json") if previous_solution else None
            )
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

    def _should_stop_early(self, critique: CriticOutput) -> bool:
        return critique.score >= self.early_stop_score and not critique.weaknesses


def record_to_schema(record: SessionRecord) -> SessionRead:
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
        history=[HistoryItem.model_validate(item) for item in (record.history or [])],
        final_answer=(
            WriterOutput.model_validate(record.final_answer)
            if record.final_answer
            else None
        ),
    )
