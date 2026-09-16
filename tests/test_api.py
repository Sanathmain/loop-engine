from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.conftest import PROBLEM


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_serves_chat_ui(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Live debate" in response.text


def test_create_get_run_session(client: TestClient) -> None:
    created = client.post("/sessions", json={"problem": PROBLEM})
    assert created.status_code == 201
    body = created.json()
    session_id = body["id"]
    assert body["problem"] == PROBLEM
    assert body["status"] == "running"
    assert body["current_round"] == 0
    assert body["history"] == []
    assert body["final_answer"] is None

    fetched = client.get(f"/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == session_id

    ran = client.post(f"/sessions/{session_id}/run")
    assert ran.status_code == 200
    result = ran.json()
    assert result["status"] == "completed"
    assert result["current_round"] == 3
    assert len(result["history"]) == 3
    assert result["final_answer"]["solution"]
    assert result["history"][0]["critic_response"]["score"] == 6.5
    assert result["history"][1]["writer_response"]["solution"]
    assert result["history"][1]["revised_writer_response"]["solution"]

    after = client.get(f"/sessions/{session_id}")
    assert after.status_code == 200
    assert after.json()["status"] == "completed"
    assert after.json()["final_answer"]["solution"] == result["final_answer"]["solution"]


def test_get_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/sessions/not-a-real-session")
    assert response.status_code == 404


def test_run_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post("/sessions/not-a-real-session/run")
    assert response.status_code == 404


def test_create_rejects_empty_problem(client: TestClient) -> None:
    response = client.post("/sessions", json={"problem": ""})
    assert response.status_code == 422


def _sse_event_types(body: str) -> list[str]:
    types: list[str] = []
    for block in body.split("\n\n"):
        data_lines = [
            line[5:].strip()
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload = json.loads("".join(data_lines))
        types.append(payload["type"])
    return types


def test_events_stream_writer_and_critic(client: TestClient) -> None:
    created = client.post("/sessions", json={"problem": PROBLEM})
    session_id = created.json()["id"]

    response = client.get(f"/sessions/{session_id}/events")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    types = _sse_event_types(response.text)
    assert "writer" in types
    assert "critic" in types
    assert types[-1] == "completed"

    stored = client.get(f"/sessions/{session_id}")
    assert stored.json()["status"] == "completed"


def test_events_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/sessions/not-a-real-session/events")
    assert response.status_code == 404
