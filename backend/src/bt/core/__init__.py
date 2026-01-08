"""
BT Core - Core runtime components for the Behavior Tree system.

Contains:
- TickContext: Execution context for tree ticks (1.2.x)
- BehaviorTree: Named tree composition with tick execution (1.3.x)
- TreeWatchdog: Detects stuck nodes based on progress tracking (1.4.x)
- TickScheduler: Manages tick execution with budget and event buffering (1.5.x)

Part of the BT Universal Runtime (spec 019).
"""

from .context import TickContext
from .tree import BehaviorTree, TreeStatus, BehaviorNodeProtocol
from .watchdog import TreeWatchdog, StuckNodeInfo
from .scheduler import TickScheduler, TickResult

__all__ = [
    # Context (1.2.x)
    "TickContext",
    # Tree (1.3.x)
    "BehaviorTree",
    "TreeStatus",
    "BehaviorNodeProtocol",
    # Watchdog (1.4.x)
    "TreeWatchdog",
    "StuckNodeInfo",
    # Scheduler (1.5.x)
    "TickScheduler",
    "TickResult",
]
