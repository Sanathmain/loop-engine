from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    problem: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    current_solution: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    history: Mapped[List[Any]] = mapped_column(JSON, nullable=False, default=list)
    final_answer: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    best_round: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
