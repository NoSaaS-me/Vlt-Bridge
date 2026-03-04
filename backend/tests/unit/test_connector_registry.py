import asyncio
import pytest
from backend.src.connectors.registry import ConnectorRegistry
from backend.src.connectors.base import BaseConnector
from backend.src.models.connectors import ConnectorAction, CredentialField, ConnectorParam


class FakeConnector(BaseConnector):
    name = "fake"
    display_name = "Fake"
    description = "Test connector"
    credential_fields = [CredentialField(name="key", label="Key")]
    actions = [
        ConnectorAction(
            name="ping",
            description="Ping",
            params=[ConnectorParam(name="msg", description="Message", required=True)],
        )
    ]

    async def invoke(self, action: str, params: dict, credentials: dict) -> dict:
        if action == "ping":
            return {"pong": params.get("msg", "")}
        raise ValueError(f"Unknown action: {action}")


def test_registry_register_and_get():
    registry = ConnectorRegistry()
    registry.register(FakeConnector)
    connector = registry.get("fake")
    assert connector is not None
    assert connector.name == "fake"


def test_registry_list_all():
    registry = ConnectorRegistry()
    registry.register(FakeConnector)
    all_connectors = registry.list_all()
    assert any(c.name == "fake" for c in all_connectors)


def test_registry_get_unknown_returns_none():
    registry = ConnectorRegistry()
    assert registry.get("nonexistent") is None


def test_connector_invoke():
    conn = FakeConnector()
    result = asyncio.run(conn.invoke("ping", {"msg": "hello"}, {}))
    assert result == {"pong": "hello"}
