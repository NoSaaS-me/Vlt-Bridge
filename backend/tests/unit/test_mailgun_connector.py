import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.src.connectors.mailgun import MailgunConnector


def test_mailgun_has_send_email_action():
    conn = MailgunConnector()
    action_names = [a.name for a in conn.actions]
    assert "send_email" in action_names


def test_mailgun_credential_fields():
    conn = MailgunConnector()
    field_names = [f.name for f in conn.credential_fields]
    assert "api_key" in field_names
    assert "domain" in field_names


@pytest.mark.asyncio
async def test_mailgun_send_email_success():
    conn = MailgunConnector()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "<msg-id>", "message": "Queued"}

    with patch("backend.src.connectors.mailgun.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await conn.invoke(
            "send_email",
            {"to": "test@example.com", "subject": "Hi", "body": "Hello"},
            {"api_key": "key-test", "domain": "mg.example.com"},
        )

    assert result["success"] is True
    assert "message_id" in result


@pytest.mark.asyncio
async def test_mailgun_unknown_action_raises():
    conn = MailgunConnector()
    with pytest.raises(ValueError, match="Unknown action"):
        await conn.invoke("bad_action", {}, {"api_key": "k", "domain": "d"})


@pytest.mark.asyncio
async def test_mailgun_missing_credentials_raises():
    conn = MailgunConnector()
    with pytest.raises(ValueError, match="api_key"):
        await conn.invoke("send_email", {"to": "a@b.com", "subject": "s", "body": "b"}, {})
