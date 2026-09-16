from __future__ import annotations

from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agents.critic import CriticAgent
from app.agents.writer import WriterAgent
from app.config import Settings
from app.database.db import get_session_factory, init_db, init_engine
from app.main import create_app
from app.models.mock_model import MockModel
from app.orchestrator.engine import Orchestrator

PROBLEM = "How can we reduce customer waiting time in our restaurant?"


@pytest.fixture
def db() -> Generator[Session, None, None]:
    init_engine("sqlite://")
    init_db()
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def orchestrator(db: Session) -> Orchestrator:
    model = MockModel()
    return Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=3,
        early_stop_score=9.0,
    )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    app = create_app(Settings(database_url="sqlite://", model_provider="mock"))
    with TestClient(app) as test_client:
        yield test_client


def make_session(orchestrator: Orchestrator, problem: str = PROBLEM) -> str:
    session_id = str(uuid4())
    orchestrator.create_session(problem=problem, session_id=session_id)
    return session_id
