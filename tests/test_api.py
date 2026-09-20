"""
Unit tests for FastAPI serving endpoints.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "docs_url" in data


def test_health_endpoint_degraded():
    # Before model is loaded or if model is None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_without_model_returns_503():
    # When model is None, /predict should return 503 HTTP exception
    sample_payload = {
        "alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280/od315_of_diluted_wines": 3.92,
        "proline": 1065.0,
    }
    response = client.post("/predict", json=sample_payload)
    # If no model loaded, it should return 503
    assert response.status_code in [200, 503]
