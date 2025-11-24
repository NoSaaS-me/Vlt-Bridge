from pathlib import Path

import pytest

from backend.src.services import config as config_module
from backend.src.services.auth import AuthError, AuthService


@pytest.fixture(autouse=True)
def restore_config_cache():
    config_module.reload_config()
    yield
    config_module.reload_config()


def test_auth_service_requires_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))

    cfg = config_module.reload_config()
    service = AuthService(config=cfg)

    with pytest.raises(AuthError) as excinfo:
        service.create_jwt("user-123")

    assert excinfo.value.error == "missing_jwt_secret"


def test_auth_service_signs_and_validates_with_secret(monkeypatch, tmp_path: Path) -> None:
    secret = "a-secure-secret-value-123"
    monkeypatch.setenv("JWT_SECRET_KEY", secret)
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))

    cfg = config_module.reload_config()
    service = AuthService(config=cfg)

    token = service.create_jwt("user-123")
    payload = service.validate_jwt(token)

    assert payload.sub == "user-123"

