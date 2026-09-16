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


class SessionCreate(BaseModel):
    problem: str = Field(min_length=1)


class SessionRead(BaseModel):
    id: str
    problem: str
    status: SessionStatus
    current_round: int
    max_rounds: int
    current_solution: WriterOutput | None = None
    history: list[HistoryItem] = Field(default_factory=list)
    final_answer: WriterOutput | None = None


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
