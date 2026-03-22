"""OracleState — LangGraph graph state for the CodeAct Oracle agent.

Extends CodeActState from langgraph-codeact. The base class already provides:
  - messages: list[BaseMessage]  — full conversation history (checkpointed)
  - context: dict[str, Any]      — REPL variable namespace (checkpointed)

Security invariant: api_key and oracle_model are NEVER stored in state.
They are passed via config["configurable"] which AsyncSqliteSaver does not persist.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from langgraph_codeact import CodeActState


class OracleState(CodeActState):
    """Graph state for the Oracle CodeAct agent.

    Inherits from CodeActState:
        messages: Annotated[list[BaseMessage], add_messages]
        context: dict[str, Any]   -- REPL namespace, auto-checkpointed

    Fields added here:
        user_id:        Scoping key — set once at thread creation.
        project_id:     Scoping key — set once at thread creation.
        plan:           Decomposed task steps from planner_node; None if bypassed.
        plan_step:      Current plan step index (0-indexed).
        recalled_facts: Fact strings loaded from Graphiti at each turn start.

    NOT stored here (security):
        api_key       — passed via config["configurable"]["api_key"]
        oracle_model  — passed via config["configurable"]["oracle_model"]
    """

    user_id: str
    project_id: str
    plan: Optional[list[str]]
    plan_step: int
    recalled_facts: list[str]
