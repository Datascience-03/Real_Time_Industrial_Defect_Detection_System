from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.app import app


def test_dashboard_is_served():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "InspectAI" in response.text
    assert "text/html" in response.headers["content-type"]


def test_dashboard_health_reports_industrial_classes():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["classes"]["0"] == "crazing"
    assert payload["classes"]["5"] == "scratches"


def test_prediction_rejects_unsupported_uploads():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a JPG, JPEG, or PNG image."


def test_prediction_accepts_valid_image():
    image_buffer = BytesIO()
    Image.new("RGB", (128, 128), color=(180, 180, 180)).save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            params={"conf": 0.25},
            files={"file": ("sample.jpg", image_buffer.getvalue(), "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.jpg"
    assert "detections" in payload
    assert "inference_time_ms" in payload
    assert payload["detection_count"] == len(payload["detections"])
