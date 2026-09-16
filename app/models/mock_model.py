from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.models.base import AIModel
from app.schemas.critic import CriticOutput
from app.schemas.writer import WriterOutput

DEFAULT_CRITIC_SCORES = [6.5, 8.0, 8.7]


def _message_text(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _extract_problem(user_text: str) -> str:
    marker = "Problem:"
    if marker not in user_text:
        return user_text.strip()
    remainder = user_text.split(marker, 1)[1]
    for stop in ("\n\nPrevious solution:", "\n\nLatest critique:", "\n\nWriter solution:"):
        if stop in remainder:
            remainder = remainder.split(stop, 1)[0]
            break
    return remainder.strip()


class MockModel(AIModel):
    """Deterministic stand-in used to exercise the Writer/Critic loop."""

    def __init__(self, critic_scores: list[float] | None = None) -> None:
        self.critic_scores = critic_scores or list(DEFAULT_CRITIC_SCORES)
        self._writer_call = 0
        self._critic_call = 0

    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        system = _message_text(messages, "system")
        user = _message_text(messages, "user")

        if response_model is WriterOutput or "You are the Writer" in system:
            return self._writer_output(user)
        if response_model is CriticOutput or "You are the Critic" in system:
            return self._critic_output(user)
        if response_model is not None:
            raise ValueError(f"MockModel has no fixture for {response_model.__name__}")
        return "Mock model response."

    def _writer_output(self, user_text: str) -> WriterOutput:
        self._writer_call += 1
        round_number = self._writer_call
        problem = _extract_problem(user_text) or "the stated problem"
        is_revision = round_number > 1

        if is_revision:
            analysis = (
                f"Round {round_number} revision of the approach to: {problem} "
                "Critique was evaluated; only changes that improve practicality were kept."
            )
            solution = (
                f"Revised plan for: {problem}. "
                "Measure the current wait (queue length, service time, arrival rate). "
                "Staff a peak-hour runner, pre-buss tables, and use a simple reservation/"
                "waitlist with SMS. Add a small holding menu for walk-ins during the rush. "
                "Review after one week using average wait and table-turn data."
            )
            assumptions = [
                "Peak congestion is concentrated in a known window.",
                "Staff can be flexibly assigned for a short trial period.",
                "Guests will use a waitlist if it is faster than standing in line.",
            ]
            risks = [
                "SMS waitlist adoption may be low for some guest segments.",
                "Extra runner labor may not pay off on slow nights.",
            ]
            confidence = min(0.55 + 0.15 * (round_number - 1), 0.9)
        else:
            analysis = (
                f"Initial analysis of: {problem} "
                "Wait time is usually a mix of arrival spikes, slow table turns, and "
                "front-of-house bottlenecks rather than kitchen speed alone."
            )
            solution = (
                f"Initial plan for: {problem}. "
                "Add a waitlist, seat parties as tables free, and add one extra server "
                "during the busiest hour."
            )
            assumptions = [
                "The main delay is seating rather than cooking.",
                "An extra server is available at peak.",
            ]
            risks = [
                "A waitlist without process change may only hide the queue.",
                "Labor cost may rise without reducing wait.",
            ]
            confidence = 0.55

        return WriterOutput(
            analysis=analysis,
            solution=solution,
            assumptions=assumptions,
            risks=risks,
            confidence=round(confidence, 2),
        )

    def _critic_output(self, user_text: str) -> CriticOutput:
        index = min(self._critic_call, len(self.critic_scores) - 1)
        score = self.critic_scores[index]
        self._critic_call += 1
        problem = _extract_problem(user_text) or "the stated problem"

        if score >= 9.0:
            return CriticOutput(
                strengths=[
                    f"The plan for '{problem}' is specific, measurable, and operationally realistic.",
                    "Assumptions are stated and the trial has a review checkpoint.",
                ],
                weaknesses=[],
                unsupported_claims=[],
                missing_information=[],
                risks=[],
                alternative_approaches=[],
                recommended_changes=[],
                score=score,
            )

        if self._critic_call == 1:
            return CriticOutput(
                strengths=[
                    "Acknowledges that waiting time has more than one cause.",
                    "Proposes a concrete, low-complexity first step.",
                ],
                weaknesses=[
                    "Treats 'add a waitlist and one server' as sufficient without measuring the actual bottleneck.",
                    "No distinction between walk-in vs reservation demand, which changes staffing math.",
                ],
                unsupported_claims=[
                    "Implies seating delay is the main cause without throughput data.",
                ],
                missing_information=[
                    "Current average wait, party-size mix, table count, and kitchen ticket times.",
                    "Labor budget and whether a runner or host is the scarce role.",
                ],
                risks=[
                    "Labor added at the wrong station increases cost with no wait reduction.",
                    "Waitlist without table-turn discipline can produce no-shows and angry guests.",
                ],
                alternative_approaches=[
                    "Time-box a one-week measurement sprint before adding headcount.",
                    "Pre-bussing and partial pre-setting to raise table turns without extra servers.",
                ],
                recommended_changes=[
                    "Collect baseline wait and ticket-time data first.",
                    "Define a peak-hour staffing experiment with a success metric (e.g. wait under 15 minutes).",
                    "Add a table-turn tactic, not only a queue tactic.",
                ],
                score=score,
            )

        return CriticOutput(
            strengths=[
                "Now includes measurement, a peak-hour experiment, and a review checkpoint.",
                "Separates queue management from table-turn work.",
            ],
            weaknesses=[
                "SMS waitlist still assumes guests will opt in; no fallback for phone-only or tourist traffic.",
                "One-week review may be too short if weekday/weekend mix is uneven.",
            ],
            unsupported_claims=[
                "Does not show why a runner is the highest-leverage extra role versus a host or expo.",
            ],
            missing_information=[
                "No numeric targets for wait time or table turn, so success is still fuzzy.",
            ],
            risks=[
                "Holding menu can slow the kitchen if not tightly scoped.",
                "Weekend patterns may reverse a weekday-only experiment.",
            ],
            alternative_approaches=[
                "Call-ahead seating or a pager/waitlist kiosk for guests who will not use SMS.",
                "Stagger reservations to flatten the arrival spike before adding labor.",
            ],
            recommended_changes=[
                "Set explicit targets (median wait, 90th-percentile wait, table turn).",
                "Name a non-SMS waitlist fallback.",
                "Run the trial across at least one weekday and one weekend peak.",
            ],
            score=score,
        )
