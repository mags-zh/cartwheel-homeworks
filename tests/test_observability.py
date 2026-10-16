"""HW2 authentication tests for the session endpoints.

Requests go through the real FastAPI routes. Every case is rejected before the
agent runs, so no Langfuse, Docker, or model provider key is needed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from server import app as server_app

client = TestClient(server_app.app)


def test_create_session_rejects_role_mismatch(world: dict) -> None:
    server_app._SESSIONS.clear()
    response = client.post("/sessions", json={"user_id": 1, "role": "support"})
    assert response.status_code == 403
    assert server_app._SESSIONS == {}


def test_token_for_one_session_cannot_authorize_another(world: dict) -> None:
    server_app._SESSIONS.clear()
    a = client.post("/sessions", json={"user_id": 1, "role": "shopper"}).json()
    b = client.post("/sessions", json={"user_id": 38, "role": "shopper"}).json()
    response = client.post(
        f"/sessions/{b['session_id']}/messages",
        json={"message": "Show my recent orders."},
        headers={"Authorization": f"Bearer {a['token']}"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "token is for another session"
