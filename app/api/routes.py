from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agents.critic import CriticAgent
from app.agents.writer import WriterAgent
from app.config import Settings, get_settings
from app.database.db import get_db
from app.models.factory import create_model
from app.orchestrator.engine import (
    Orchestrator,
    SessionContinueError,
    SessionNotFoundError,
    record_to_schema,
)
from app.schemas.session import SessionContinue, SessionCreate, SessionRead, SessionSummary

router = APIRouter()


def get_app_settings(request: Request) -> Settings:
    stored = getattr(request.app.state, "settings", None)
    if stored is not None:
        return stored
    return get_settings()


def get_orchestrator(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> Orchestrator:
    return Orchestrator(
        db=db,
        max_rounds=settings.max_rounds,
        early_stop_score=settings.early_stop_score,
        improvement_epsilon=settings.improvement_epsilon,
    )


def get_running_orchestrator(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> Orchestrator:
    model = create_model(settings)
    return Orchestrator(
        db=db,
        writer=WriterAgent(model=model),
        critic=CriticAgent(model=model),
        max_rounds=settings.max_rounds,
        early_stop_score=settings.early_stop_score,
        improvement_epsilon=settings.improvement_epsilon,
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> list[SessionSummary]:
    return orchestrator.list_sessions()


@router.post("/sessions", response_model=SessionRead, status_code=201)
def create_session(
    payload: SessionCreate,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> SessionRead:
    record = orchestrator.create_session(
        problem=payload.problem,
        session_id=str(uuid4()),
        max_rounds=payload.max_rounds,
    )
    return record_to_schema(record)


@router.get("/sessions/{session_id}", response_model=SessionRead)
def get_session(
    session_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> SessionRead:
    try:
        record = orchestrator.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record_to_schema(record)


@router.post("/sessions/{session_id}/run", response_model=SessionRead)
async def run_session(
    session_id: str,
    orchestrator: Orchestrator = Depends(get_running_orchestrator),
) -> SessionRead:
    try:
        return await orchestrator.run(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Loop failed: {exc}",
        ) from exc


@router.post("/sessions/{session_id}/continue", response_model=SessionRead)
def continue_session(
    session_id: str,
    payload: SessionContinue | None = None,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> SessionRead:
    body = payload or SessionContinue()
    try:
        record = orchestrator.prepare_continue(
            session_id=session_id,
            extra_rounds=body.extra_rounds,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionContinueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record_to_schema(record)


@router.get("/sessions/{session_id}/events")
async def stream_session_events(
    session_id: str,
    mode: str | None = None,
    orchestrator: Orchestrator = Depends(get_running_orchestrator),
) -> StreamingResponse:
    try:
        orchestrator.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    user_unsatisfied = mode == "continue"

    async def event_stream() -> AsyncIterator[str]:
        async for event in orchestrator.iter_run(
            session_id,
            user_unsatisfied=user_unsatisfied,
        ):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
