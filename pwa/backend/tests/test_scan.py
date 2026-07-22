import pytest


def test_scan_requires_auth(client, sample_qr_image_bytes):
    response = client.post(
        "/api/v1/scan",
        files={"file": ("qr.png", sample_qr_image_bytes, "image/png")},
    )
    assert response.status_code == 401


def test_scan_rejects_unsupported_content_type(client, auth_headers):
    response = client.post(
        "/api/v1/scan",
        headers=auth_headers,
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_scan_rejects_mislabeled_file(client, auth_headers):
    # Declares image/png but the bytes aren't actually a PNG - the
    # magic-byte sniff in validate_and_read_upload must catch this.
    response = client.post(
        "/api/v1/scan",
        headers=auth_headers,
        files={"file": ("fake.png", b"not really a png", "image/png")},
    )
    assert response.status_code == 400


def test_scan_full_pipeline(client, auth_headers, sample_qr_image_bytes, engine_available):
    if not engine_available:
        pytest.skip("QR Shield engine (src.*) not importable in this environment")

    response = client.post(
        "/api/v1/scan",
        headers=auth_headers,
        files={"file": ("qr.png", sample_qr_image_bytes, "image/png")},
    )
    assert response.status_code == 201

    body = response.json()
    assert body["scan_id"].startswith("scan_")
    assert body["verdict"] in ("safe", "suspicious", "tampered", "no_qr_found")

    # QR Detection stage
    assert body["qr"]["decoded"] is True
    assert body["qr"]["data"] == "https://example.com/test-page"

    # Tamper Analysis stage - reused engine, not a duplicate implementation
    assert body["tamper"]["engine"] == "qr_shield_core.tamper_analysis.TamperDetector"

    # URL Analysis stage (Change 5 - previously always the default/unset placeholder)
    assert body["url_analysis"]["analyzed"] is True
    assert body["url_analysis"]["url"] == "https://example.com/test-page"

    # Risk Assessment stage (Change 5 - previously always the default/unset placeholder)
    assert body["risk_assessment"]["assessed"] is True
    assert body["risk_assessment"]["risk_level"] in ("SAFE", "SUSPICIOUS", "HIGH_RISK")

    # Top-level fields are now actually populated from the risk engine
    assert body["recommendation"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["processing_times"]["total_ms"] > 0
