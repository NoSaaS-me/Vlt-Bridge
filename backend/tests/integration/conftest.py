"""Integration test conftest — allow outbound network for live API tests."""
import pytest


def pytest_configure(config):
    """Register integration marker."""
    config.addinivalue_line(
        "markers", "live: tests that make real outbound API calls (LLM, etc.)"
    )


@pytest.fixture(autouse=True)
def allow_network(socket_enabled):
    """Allow all network access in integration tests."""
    pass
