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
    assert "Past sessions" in response.text
    assert "Starting rounds" in response.text
    assert 'id="max-rounds"' in response.text
    assert "Continue 3 more rounds" in response.text


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
    assert result["best_round"] == 3
    assert result["history"][0]["critic_response"]["score"] == 6.5
    assert result["history"][1]["writer_response"]["solution"]
    assert result["history"][1]["revised_writer_response"]["solution"]

    after = client.get(f"/sessions/{session_id}")
    assert after.status_code == 200
    assert after.json()["status"] == "completed"
    assert after.json()["final_answer"]["solution"] == result["final_answer"]["solution"]


def test_list_sessions(client: TestClient) -> None:
    created = client.post("/sessions", json={"problem": PROBLEM})
    session_id = created.json()["id"]
    client.post(f"/sessions/{session_id}/run")

    listed = client.get("/sessions")
    assert listed.status_code == 200
    rows = listed.json()
    assert isinstance(rows, list)
    assert any(row["id"] == session_id for row in rows)
    match = next(row for row in rows if row["id"] == session_id)
    assert match["best_score"] == 8.7
    assert match["best_round"] == 3
    assert match["status"] == "completed"


def test_get_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/sessions/not-a-real-session")
    assert response.status_code == 404


def test_run_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post("/sessions/not-a-real-session/run")
    assert response.status_code == 404


def test_create_rejects_empty_problem(client: TestClient) -> None:
    response = client.post("/sessions", json={"problem": ""})
    assert response.status_code == 422


def test_create_with_custom_max_rounds(client: TestClient) -> None:
    created = client.post(
        "/sessions",
        json={"problem": PROBLEM, "max_rounds": 5},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["max_rounds"] == 5

    ran = client.post(f"/sessions/{body['id']}/run")
    assert ran.status_code == 200
    result = ran.json()
    assert result["max_rounds"] == 5
    assert result["current_round"] == 5
    assert len(result["history"]) == 5


def test_create_rejects_invalid_max_rounds(client: TestClient) -> None:
    too_low = client.post("/sessions", json={"problem": PROBLEM, "max_rounds": 0})
    assert too_low.status_code == 422
    too_high = client.post("/sessions", json={"problem": PROBLEM, "max_rounds": 11})
    assert too_high.status_code == 422


def test_continue_session_runs_more_rounds(client: TestClient) -> None:
    created = client.post(
        "/sessions",
        json={"problem": PROBLEM, "max_rounds": 2},
    )
    session_id = created.json()["id"]
    first = client.post(f"/sessions/{session_id}/run")
    assert first.status_code == 200
    assert first.json()["current_round"] == 2
    assert first.json()["status"] == "completed"

    continued = client.post(
        f"/sessions/{session_id}/continue",
        json={"extra_rounds": 2},
    )
    assert continued.status_code == 200
    body = continued.json()
    assert body["status"] == "running"
    assert body["max_rounds"] == 4
    assert body["final_answer"] is None

    again = client.post(f"/sessions/{session_id}/run")
    assert again.status_code == 200
    result = again.json()
    assert result["status"] == "completed"
    assert result["current_round"] == 4
    assert len(result["history"]) == 4


def test_continue_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/sessions/not-a-real-session/continue",
        json={"extra_rounds": 2},
    )
    assert response.status_code == 404


def test_continue_without_rounds_returns_400(client: TestClient) -> None:
    created = client.post("/sessions", json={"problem": PROBLEM})
    session_id = created.json()["id"]
    response = client.post(
        f"/sessions/{session_id}/continue",
        json={"extra_rounds": 2},
    )
    assert response.status_code == 400



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
