from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, get_settings
from app.database.db import init_db, init_engine

STATIC_DIR = Path(__file__).resolve().parent / "static"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        init_engine(settings.database_url)
        init_db()
        yield

    application = FastAPI(
        title="Loop Engine",
        description="Writer/Critic multi-agent reasoning loop",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return application


app = create_app()
