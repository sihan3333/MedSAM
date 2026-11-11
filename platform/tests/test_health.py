from fastapi.testclient import TestClient

from annotation_app.main import create_app
from annotation_app.settings import Settings


def test_health_and_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(data_root=tmp_path)
    assert settings.database_url.endswith("/medsam")
    assert settings.device == "cpu"
    response = TestClient(create_app(settings)).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
