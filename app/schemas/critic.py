from __future__ import annotations

from pydantic import BaseModel, Field


class CriticOutput(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    unsupported_claims: list[str]
    missing_information: list[str]
    risks: list[str]
    alternative_approaches: list[str]
    recommended_changes: list[str]
    score: float = Field(ge=0.0, le=10.0)
