"""Artifact event bus for IPC between artifacts."""

import asyncio
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

MAX_HOP_COUNT = 10  # Prevent infinite event loops


class ArtifactEventBus:
    """Cross-artifact event routing through the daemon."""

    def __init__(self):
        # artifact_id -> list of (event_type, callback)
        self._subscriptions: dict[str, list[tuple[str, Callable]]] = {}

    async def emit(
        self,
        source_artifact: str,
        event_type: str,
        payload: dict,
        _hop: int = 0,
    ) -> list[str]:
        """Route event to all subscribers. Returns list of recipient artifact IDs."""
        if _hop >= MAX_HOP_COUNT:
            log.warning(f"Event hop limit reached for {event_type} from {source_artifact}")
            return []

        recipients = []
        for artifact_id, subs in self._subscriptions.items():
            if artifact_id == source_artifact:
                continue  # No self-notification
            for sub_event, callback in subs:
                if sub_event == event_type or sub_event == "*":
                    try:
                        await callback(artifact_id, event_type, source_artifact, payload)
                        recipients.append(artifact_id)
                    except Exception as e:
                        log.error(f"Event handler error for {artifact_id}: {e}")

        return list(set(recipients))  # deduplicate

    def subscribe(self, artifact_id: str, event_type: str, callback: Callable):
        """Register a subscription."""
        self._subscriptions.setdefault(artifact_id, []).append((event_type, callback))

    def unsubscribe(self, artifact_id: str):
        """Remove all subscriptions for an artifact."""
        self._subscriptions.pop(artifact_id, None)

    def get_subscriptions(self, artifact_id: str) -> list[str]:
        """Get event types an artifact subscribes to."""
        return [et for et, _ in self._subscriptions.get(artifact_id, [])]


# Singleton
_event_bus: Optional[ArtifactEventBus] = None


def get_event_bus() -> ArtifactEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = ArtifactEventBus()
    return _event_bus
