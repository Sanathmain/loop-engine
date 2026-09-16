from __future__ import annotations

from app.config import get_settings
from app.database.db import get_engine, init_db, init_engine
from app.database.models import Base


def create_database() -> str:
    """Create the SQLite file and all tables if they do not already exist."""
    settings = get_settings()
    init_engine(settings.database_url)
    init_db()
    return settings.database_url


def table_names() -> list[str]:
    return sorted(Base.metadata.tables.keys())
