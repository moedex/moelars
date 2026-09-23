import pytest
from fastapi.testclient import TestClient

from moelars.backends.mock import MockBackend
from moelars.engine import Engine
from moelars.server import create_app

BODY = {
    "model": "moelars-latest",
    "state": "I was charged twice for my subscription.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
        },
    },
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MOELARS_API_KEY", raising=False)
    return TestClient(create_app(Engine(MockBackend())))


def test_systemone_wire_shape(client):
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"model", "answers", "usage"}
    assert data["answers"]["refund"].keys() == {"type", "noul"}
    assert set(data["answers"]["department"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(data["usage"]) == {"input_tokens", "output_tokens"}
    assert response.headers["x-typesafe-request-id"]


def test_extensions_are_opt_in(client):
    body = {**BODY, "moelars": {"permutations": 2, "abstain_margin": 0.1}}
    data = client.post("/v1/systemone", json=body).json()
    assert "order_sensitivity" in data["answers"]["department"]
    assert "abstain" in data["answers"]["department"]


def test_validation_error_shape(client):
    bad = {**BODY, "questions": {"x": {"type": "verdict", "instructions": "?"}}}
    response = client.post("/v1/systemone", json=bad)
    assert response.status_code == 422
    assert set(response.json()) == {"message", "error_type"}
    assert response.json()["error_type"] == "invalid_request"


def test_models_endpoint(client):
    data = client.get("/v1/models").json()
    names = [m["name"] for m in data["models"]]
    assert "moelars-latest" in names
    assert all({"name", "description", "release_date"} <= set(m) for m in data["models"])


def test_auth_when_key_configured(monkeypatch):
    monkeypatch.setenv("MOELARS_API_KEY", "secret")
    client = TestClient(create_app(Engine(MockBackend())))
    assert client.post("/v1/systemone", json=BODY).status_code == 401
    ok = client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert client.get("/healthz").status_code == 200
