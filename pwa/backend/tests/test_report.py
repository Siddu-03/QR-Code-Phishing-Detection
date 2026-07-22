import pytest


def _create_scan(client, auth_headers, sample_qr_image_bytes) -> str:
    response = client.post(
        "/api/v1/scan",
        headers=auth_headers,
        files={"file": ("qr.png", sample_qr_image_bytes, "image/png")},
    )
    assert response.status_code == 201
    return response.json()["scan_id"]


def test_generate_report_requires_auth(client):
    response = client.post("/api/v1/report", json={"scan_id": "scan_does_not_exist"})
    assert response.status_code == 401


def test_download_report_requires_auth(client):
    # This endpoint had no auth at all before Change 2.
    response = client.get("/api/v1/report/scan_does_not_exist/download")
    assert response.status_code == 401


def test_generate_report_for_unknown_scan_returns_404(client, auth_headers):
    response = client.post(
        "/api/v1/report", headers=auth_headers, json={"scan_id": "scan_does_not_exist"}
    )
    assert response.status_code == 404


def test_download_report_for_unknown_scan_returns_404(client, auth_headers):
    response = client.get("/api/v1/report/scan_does_not_exist/download", headers=auth_headers)
    assert response.status_code == 404


def test_generate_and_download_report(client, auth_headers, sample_qr_image_bytes, engine_available):
    if not engine_available:
        pytest.skip("QR Shield engine (src.*) not importable in this environment")

    scan_id = _create_scan(client, auth_headers, sample_qr_image_bytes)

    gen_response = client.post(
        "/api/v1/report", headers=auth_headers, json={"scan_id": scan_id, "format": "json"}
    )
    assert gen_response.status_code == 200
    assert gen_response.json()["scan_id"] == scan_id

    dl_response = client.get(
        f"/api/v1/report/{scan_id}/download", headers=auth_headers, params={"format": "json"}
    )
    assert dl_response.status_code == 200
    assert dl_response.json()["scan_id"] == scan_id


def test_download_report_rejects_bad_format(client, auth_headers, sample_qr_image_bytes, engine_available):
    if not engine_available:
        pytest.skip("QR Shield engine (src.*) not importable in this environment")

    scan_id = _create_scan(client, auth_headers, sample_qr_image_bytes)

    response = client.get(
        f"/api/v1/report/{scan_id}/download", headers=auth_headers, params={"format": "docx"}
    )
    assert response.status_code == 400
