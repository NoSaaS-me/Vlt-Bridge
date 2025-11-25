from pathlib import Path

import pytest

from backend.src.services.config import AppConfig
from backend.src.services.vault import VaultService, sanitize_path


@pytest.fixture
def vault_config(tmp_path: Path) -> AppConfig:
    return AppConfig(vault_base_path=tmp_path / "vaults")


def test_initialize_vault_creates_user_directory(vault_config: AppConfig) -> None:
    service = VaultService(config=vault_config)

    path = service.initialize_vault("user-123")

    expected = vault_config.vault_base_path / "user-123"
    assert path == expected.resolve()
    assert path.exists() and path.is_dir()


def test_sanitize_path_blocks_escape(vault_config: AppConfig, tmp_path: Path) -> None:
    base = vault_config.vault_base_path

    with pytest.raises(ValueError):
        sanitize_path("user-123", base, "../outside.md")


def test_write_and_read_note_round_trip(vault_config: AppConfig) -> None:
    service = VaultService(config=vault_config)

    service.write_note(
        "user-123",
        "folder/note.md",
        title="My Note",
        metadata={"tags": ["dev", "docs"]},
        body="Hello World",
    )

    note = service.read_note("user-123", "folder/note.md")

    assert note["path"] == "folder/note.md"
    assert note["body"] == "Hello World"
    assert note["metadata"]["title"] == "My Note"
    assert note["absolute_path"].is_absolute()
    assert str(note["absolute_path"]).startswith(str(vault_config.vault_base_path))
