def test_health_returns_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"]
    assert body["version"]
    assert "timestamp" in body


def test_health_does_not_require_auth(client):
    # No X-API-Key header supplied - health must stay publicly reachable
    # so the frontend/monitoring can check availability before login.
    response = client.get("/api/v1/health")
    assert response.status_code == 200
