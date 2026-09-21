from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.critic import CriticOutput
from app.schemas.writer import WriterOutput


class SessionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HistoryItem(BaseModel):
    round_number: int
    writer_response: WriterOutput
    critic_response: CriticOutput
    revised_writer_response: WriterOutput | None = None
    timestamp: datetime
    duration_ms: int | None = None
    tokens: int | None = None


class SessionCreate(BaseModel):
    problem: str = Field(min_length=1)
    max_rounds: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional per-session round limit. Defaults to server MAX_ROUNDS.",
    )


class SessionContinue(BaseModel):
    extra_rounds: int = Field(
        default=2,
        ge=1,
        le=10,
        description="How many additional Writer/Critic rounds to run.",
    )


class SessionRead(BaseModel):
    id: str
    problem: str
    status: SessionStatus
    current_round: int
    max_rounds: int
    current_solution: WriterOutput | None = None
    history: list[HistoryItem] = Field(default_factory=list)
    final_answer: WriterOutput | None = None
    best_round: int | None = None
    created_at: datetime | None = None


class SessionSummary(BaseModel):
    id: str
    problem: str
    status: SessionStatus
    current_round: int
    max_rounds: int
    best_score: float | None = None
    best_round: int | None = None
    created_at: datetime | None = None


class LoopEventType(str, Enum):
    ROUND = "round"
    WRITER = "writer"
    CRITIC = "critic"
    COMPLETED = "completed"
    ERROR = "error"


class LoopEvent(BaseModel):
    type: LoopEventType
    round_number: int | None = None
    payload: dict[str, Any] | None = None
