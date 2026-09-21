from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class WriterOutput(BaseModel):
    analysis: str
    solution: str
    assumptions: list[str]
    risks: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence from 0.0 to 1.0 (not a percentage).",
    )
    confidence_rationale: str = Field(
        default="",
        description="Why this confidence is justified given known gaps.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        number = float(value)
        if number > 1.0:
            number = number / 100.0
        return max(0.0, min(number, 1.0))
