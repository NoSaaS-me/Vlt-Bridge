from pathlib import Path

import pytest

from backend.src.services import config as config_module


@pytest.fixture(autouse=True)
def restore_config_cache():
    """
    Ensure configuration cache is cleared between tests.
    """
    config_module.reload_config()
    yield
    config_module.reload_config()


def test_get_config_allows_missing_jwt_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))

    cfg = config_module.reload_config()

    assert cfg.jwt_secret_key is None
    assert cfg.vault_base_path == tmp_path.resolve()


def test_get_config_rejects_short_jwt_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VAULT_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("JWT_SECRET_KEY", "short")

    with pytest.raises(ValueError):
        config_module.reload_config()

