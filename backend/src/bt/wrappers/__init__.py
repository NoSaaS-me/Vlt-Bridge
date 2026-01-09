"""
BT Wrappers - Bridge classes between BT trees and external interfaces.

This package provides wrapper classes that adapt behavior trees to
work with existing service interfaces like SSE streaming.

Part of the BT Universal Runtime (spec 019).
Updated for 020-bt-oracle-agent: Direct BT routing without shadow mode.
"""

from .oracle_wrapper import OracleBTWrapper, OracleStreamChunk, create_oracle_bt_wrapper
from .research_wrapper import (
    ResearchBTWrapper,
    ResearchProgressChunk,
    ResearchCompleteChunk,
    create_research_bt_wrapper,
)

__all__ = [
    # Oracle wrapper
    "OracleBTWrapper",
    "OracleStreamChunk",
    "create_oracle_bt_wrapper",
    # Research wrapper
    "ResearchBTWrapper",
    "ResearchProgressChunk",
    "ResearchCompleteChunk",
    "create_research_bt_wrapper",
]
