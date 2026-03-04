"""Tests for ConnectorService: per-user connector configuration CRUD."""
import pytest
from backend.src.services.connector_service import ConnectorService
from backend.src.services.database import DatabaseService


@pytest.fixture
def db(tmp_path):
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
    db = DatabaseService(db_path=tmp_path / "test.db")
    db.initialize()
    return db


@pytest.fixture
def svc(db):
    return ConnectorService(db_service=db)


def test_get_config_empty(svc):
    config = svc.get_config("user1", "mailgun")
    assert config == {}


def test_set_and_get_config(svc):
    svc.set_config("user1", "mailgun", {"api_key": "secret-key", "domain": "mg.test.com"})
    config = svc.get_config("user1", "mailgun")
    assert config["api_key"] == "secret-key"
    assert config["domain"] == "mg.test.com"


def test_is_enabled_default_false(svc):
    assert svc.is_enabled("user1", "mailgun") is False


def test_set_enabled(svc):
    svc.set_config("user1", "mailgun", {"__enabled": "true"})
    assert svc.is_enabled("user1", "mailgun") is True


def test_get_credentials_excludes_enabled_key(svc):
    svc.set_config("user1", "mailgun", {"api_key": "k", "__enabled": "true"})
    creds = svc.get_credentials("user1", "mailgun")
    assert "api_key" in creds
    assert "__enabled" not in creds


def test_is_configured_requires_all_required_fields(svc):
    from backend.src.connectors.mailgun import MailgunConnector
    conn = MailgunConnector()
    # No credentials set — not configured
    assert svc.is_configured("user1", conn) is False
    # Set only required (secret) fields
    svc.set_config("user1", "mailgun", {"api_key": "k", "domain": "mg.test.com"})
    assert svc.is_configured("user1", conn) is True
