"""Unit tests for artifact service — state machine, CRUD, manifest validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vlt.core.models import ArtifactState, ARTIFACT_STATE_TRANSITIONS


# ---------------------------------------------------------------------------
# State Machine Tests
# ---------------------------------------------------------------------------

class TestArtifactStateMachine:
    """Test the state transition graph validation."""

    def test_valid_transitions_from_draft(self):
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.DRAFT]
        assert ArtifactState.BUILDING in valid
        assert len(valid) == 1  # Draft can only go to building

    def test_valid_transitions_from_building(self):
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.BUILDING]
        assert ArtifactState.TESTING in valid
        assert ArtifactState.ERROR in valid

    def test_valid_transitions_from_testing(self):
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.TESTING]
        assert ArtifactState.REVIEWING in valid
        assert ArtifactState.APPROVED in valid
        assert ArtifactState.BUILDING in valid
        assert ArtifactState.ERROR in valid

    def test_valid_transitions_from_approved(self):
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.APPROVED]
        assert ArtifactState.TESTING in valid  # code change demotion
        assert ArtifactState.DEPLOYED in valid
        assert ArtifactState.BUILDING in valid

    def test_valid_transitions_from_deployed(self):
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.DEPLOYED]
        assert ArtifactState.UPDATE_PENDING in valid
        assert ArtifactState.BUILDING in valid
        assert ArtifactState.ERROR in valid

    def test_valid_transitions_from_error(self):
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.ERROR]
        assert ArtifactState.BUILDING in valid
        assert ArtifactState.DRAFT in valid

    def test_invalid_transition_draft_to_approved(self):
        """Cannot skip from draft directly to approved."""
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.DRAFT]
        assert ArtifactState.APPROVED not in valid

    def test_invalid_transition_draft_to_deployed(self):
        """Cannot skip from draft directly to deployed."""
        valid = ARTIFACT_STATE_TRANSITIONS[ArtifactState.DRAFT]
        assert ArtifactState.DEPLOYED not in valid

    def test_all_states_have_transitions(self):
        """Every state must have at least one valid transition."""
        for state in ArtifactState:
            assert state in ARTIFACT_STATE_TRANSITIONS, f"{state} missing from transitions"
            assert len(ARTIFACT_STATE_TRANSITIONS[state]) > 0, f"{state} has no transitions"


# ---------------------------------------------------------------------------
# Artifact Event Bus Tests
# ---------------------------------------------------------------------------

class TestArtifactEventBus:
    """Test the artifact event bus IPC system."""

    def test_emit_to_subscriber(self):
        from vlt.daemon.artifact_event_bus import ArtifactEventBus
        import asyncio

        bus = ArtifactEventBus()
        received = []

        async def handler(recipient, event_type, source, payload):
            received.append({"recipient": recipient, "type": event_type, "source": source, "payload": payload})

        bus.subscribe("artifact-b", "test_event", handler)
        recipients = asyncio.new_event_loop().run_until_complete(
            bus.emit("artifact-a", "test_event", {"key": "value"})
        )

        assert "artifact-b" in recipients
        assert len(received) == 1
        assert received[0]["type"] == "test_event"
        assert received[0]["source"] == "artifact-a"

    def test_no_self_notification(self):
        from vlt.daemon.artifact_event_bus import ArtifactEventBus
        import asyncio

        bus = ArtifactEventBus()
        received = []

        async def handler(recipient, event_type, source, payload):
            received.append(recipient)

        bus.subscribe("artifact-a", "test_event", handler)
        recipients = asyncio.new_event_loop().run_until_complete(
            bus.emit("artifact-a", "test_event", {})
        )

        assert len(received) == 0
        assert len(recipients) == 0

    def test_unsubscribe(self):
        from vlt.daemon.artifact_event_bus import ArtifactEventBus
        import asyncio

        bus = ArtifactEventBus()
        received = []

        async def handler(recipient, event_type, source, payload):
            received.append(True)

        bus.subscribe("artifact-b", "test_event", handler)
        bus.unsubscribe("artifact-b")

        recipients = asyncio.new_event_loop().run_until_complete(
            bus.emit("artifact-a", "test_event", {})
        )

        assert len(received) == 0

    def test_hop_limit(self):
        from vlt.daemon.artifact_event_bus import ArtifactEventBus, MAX_HOP_COUNT
        import asyncio

        bus = ArtifactEventBus()

        async def handler(recipient, event_type, source, payload):
            pass

        bus.subscribe("artifact-b", "test", handler)
        # Simulate exceeding hop limit
        recipients = asyncio.new_event_loop().run_until_complete(
            bus.emit("artifact-a", "test", {}, _hop=MAX_HOP_COUNT)
        )
        assert len(recipients) == 0


# ---------------------------------------------------------------------------
# Harness Tests
# ---------------------------------------------------------------------------

class TestArtifactHarness:
    """Test the artifact backend harness module."""

    def test_load_artifact_module(self):
        from vlt.daemon.artifact_harness import load_artifact_module

        with tempfile.TemporaryDirectory() as tmpdir:
            main_py = Path(tmpdir) / "main.py"
            main_py.write_text(
                "def handle(action, params):\n"
                "    return {'echo': action}\n"
            )
            module = load_artifact_module(tmpdir)
            assert hasattr(module, "handle")
            result = module.handle("test", {})
            assert result == {"echo": "test"}

    def test_load_missing_module(self):
        from vlt.daemon.artifact_harness import load_artifact_module

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_artifact_module(tmpdir)

    def test_state_functions(self):
        from vlt.daemon.artifact_harness import load_artifact_module

        with tempfile.TemporaryDirectory() as tmpdir:
            main_py = Path(tmpdir) / "main.py"
            main_py.write_text(
                "_state = {}\n"
                "def handle(action, params): return {'ok': True}\n"
                "def save_state(): return {'counter': 42}\n"
                "def load_state(s): _state.update(s)\n"
            )
            module = load_artifact_module(tmpdir)
            assert hasattr(module, "save_state")
            assert hasattr(module, "load_state")
            state = module.save_state()
            assert state == {"counter": 42}


# ---------------------------------------------------------------------------
# Watcher Classification Tests
# ---------------------------------------------------------------------------

class TestArtifactWatcher:
    """Test file change classification logic."""

    def test_classify_css_only(self):
        from vlt.daemon.artifact_watcher import ArtifactWatcher
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = ArtifactWatcher("test", tmpdir, asyncio.new_event_loop())
            event = watcher._classify_changes({
                f"{tmpdir}/frontend/style.css",
                f"{tmpdir}/frontend/theme.css",
            })
            assert event.css_only is True
            assert event.frontend_changed is True
            assert event.backend_changed is False

    def test_classify_js_change(self):
        from vlt.daemon.artifact_watcher import ArtifactWatcher
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = ArtifactWatcher("test", tmpdir, asyncio.new_event_loop())
            event = watcher._classify_changes({
                f"{tmpdir}/frontend/app.js",
            })
            assert event.css_only is False
            assert event.frontend_changed is True

    def test_classify_backend_change(self):
        from vlt.daemon.artifact_watcher import ArtifactWatcher
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = ArtifactWatcher("test", tmpdir, asyncio.new_event_loop())
            event = watcher._classify_changes({
                f"{tmpdir}/backend/main.py",
            })
            assert event.backend_changed is True
            assert event.frontend_changed is False
            assert event.css_only is False

    def test_classify_manifest_change(self):
        from vlt.daemon.artifact_watcher import ArtifactWatcher
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = ArtifactWatcher("test", tmpdir, asyncio.new_event_loop())
            event = watcher._classify_changes({
                f"{tmpdir}/manifest.json",
            })
            assert event.manifest_changed is True

    def test_classify_mixed_changes(self):
        from vlt.daemon.artifact_watcher import ArtifactWatcher
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = ArtifactWatcher("test", tmpdir, asyncio.new_event_loop())
            event = watcher._classify_changes({
                f"{tmpdir}/frontend/style.css",
                f"{tmpdir}/frontend/app.js",
                f"{tmpdir}/backend/main.py",
            })
            assert event.css_only is False  # Has non-CSS frontend changes
            assert event.frontend_changed is True
            assert event.backend_changed is True
