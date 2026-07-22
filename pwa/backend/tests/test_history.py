import pytest


def test_history_requires_auth(client):
    # This endpoint had no auth at all before Change 2.
    response = client.get("/api/v1/history")
    assert response.status_code == 401


def test_history_rejects_bad_key(client):
    response = client.get("/api/v1/history", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_history_lists_scans(client, auth_headers, sample_qr_image_bytes, engine_available):
    if not engine_available:
        pytest.skip("QR Shield engine (src.*) not importable in this environment")

    scan_response = client.post(
        "/api/v1/scan",
        headers=auth_headers,
        files={"file": ("qr.png", sample_qr_image_bytes, "image/png")},
    )
    assert scan_response.status_code == 201
    scan_id = scan_response.json()["scan_id"]

    history_response = client.get("/api/v1/history", headers=auth_headers)
    assert history_response.status_code == 200

    body = history_response.json()
    assert body["total"] >= 1
    assert any(item["scan_id"] == scan_id for item in body["items"])


def test_history_pagination_params_are_validated(client, auth_headers):
    response = client.get("/api/v1/history", headers=auth_headers, params={"limit": 0})
    assert response.status_code == 422
