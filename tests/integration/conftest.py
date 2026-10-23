import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import audit, service


@pytest.fixture
def client(trained, tmp_path):
    model, info = trained
    service.set_model(model, info)
    audit.configure(tmp_path / "audit.db")
    yield TestClient(app)
    service.set_model(None)
