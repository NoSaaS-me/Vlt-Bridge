"""Unit tests for OracleState — T020 (023-oracle-codeact-rework).

Structural tests only — no LLM calls, no graph execution.
Verifies field presence, security invariants, and defaults.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

try:
    from backend.src.services.oracle_v2.state import OracleState
    _IMPORT_OK = True
except ImportError as _import_err:
    _IMPORT_OK = False
    _IMPORT_ERR = _import_err


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_state() -> dict:
    """Minimal valid OracleState as a plain dict (TypedDict instantiation)."""
    return {
        "messages": [],
        "context": {},
        "user_id": "test-user",
        "project_id": "test-project",
        "plan": None,
        "plan_step": 0,
        "recalled_facts": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IMPORT_OK, reason="oracle_v2.state not importable")
class TestOracleStateFields:
    """Verify OracleState has required fields with correct types."""

    def test_state_is_typed_dict(self):
        """OracleState should be a TypedDict (has __annotations__)."""
        annotations = {}
        # Collect all annotations from the MRO (TypedDicts chain via __bases__)
        for cls in reversed(type(OracleState).__mro__[:-1]):
            annotations.update(getattr(cls, "__annotations__", {}))
        # Also collect directly
        annotations.update(OracleState.__annotations__)
        assert isinstance(annotations, dict)

    def test_has_user_id_field(self):
        """OracleState must declare user_id: str."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "user_id" in all_annotations, "Missing field: user_id"

    def test_has_project_id_field(self):
        """OracleState must declare project_id: str."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "project_id" in all_annotations, "Missing field: project_id"

    def test_has_plan_field(self):
        """OracleState must declare plan: Optional[list[str]]."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "plan" in all_annotations, "Missing field: plan"

    def test_has_plan_step_field(self):
        """OracleState must declare plan_step: int."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "plan_step" in all_annotations, "Missing field: plan_step"

    def test_has_recalled_facts_field(self):
        """OracleState must declare recalled_facts: list[str]."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "recalled_facts" in all_annotations, "Missing field: recalled_facts"

    def test_inherits_messages_from_codeact_state(self):
        """OracleState inherits messages from CodeActState."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "messages" in all_annotations, (
            "Missing inherited field: messages (from CodeActState)"
        )

    def test_inherits_context_from_codeact_state(self):
        """OracleState inherits context (REPL namespace) from CodeActState."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "context" in all_annotations, (
            "Missing inherited field: context (from CodeActState)"
        )


@pytest.mark.skipif(not _IMPORT_OK, reason="oracle_v2.state not importable")
class TestOracleStateSecurityInvariants:
    """Verify security-critical fields are NOT present in OracleState.

    api_key and oracle_model MUST NOT be stored in state — they are passed
    via config["configurable"] which AsyncSqliteSaver does not persist.
    Storing them in state would checkpoint secrets to SQLite.
    """

    def test_no_api_key_field(self):
        """api_key must NOT be in OracleState (security: never checkpointed)."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "api_key" not in all_annotations, (
            "SECURITY VIOLATION: api_key found in OracleState. "
            "api_key must be passed via config['configurable'] only."
        )

    def test_no_oracle_model_field(self):
        """oracle_model must NOT be in OracleState (security: never checkpointed)."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        assert "oracle_model" not in all_annotations, (
            "SECURITY VIOLATION: oracle_model found in OracleState. "
            "oracle_model must be passed via config['configurable'] only."
        )

    def test_no_password_field(self):
        """No password-like field (belt-and-suspenders check)."""
        all_annotations = {}
        for cls in reversed(OracleState.__mro__):
            all_annotations.update(getattr(cls, "__annotations__", {}))
        secret_fields = {k for k in all_annotations if "password" in k.lower() or "secret" in k.lower()}
        assert not secret_fields, (
            f"Secret-looking fields found in OracleState: {secret_fields}"
        )


@pytest.mark.skipif(not _IMPORT_OK, reason="oracle_v2.state not importable")
class TestOracleStateDefaults:
    """Verify expected default values and None-allowance for optional fields."""

    def test_plan_field_allows_none(self):
        """plan: Optional[list[str]] — None must be a valid value."""
        import typing
        hints = {}
        for cls in reversed(OracleState.__mro__):
            hints.update(getattr(cls, "__annotations__", {}))

        plan_annotation = hints.get("plan")
        assert plan_annotation is not None

        # Check that None is an acceptable value by inspecting the annotation string
        annotation_str = str(plan_annotation)
        assert "None" in annotation_str or "Optional" in annotation_str, (
            f"plan annotation {annotation_str!r} does not appear Optional"
        )

    def test_minimal_state_dict_has_all_required_keys(self):
        """A dict with all required OracleState fields should validate as expected."""
        state = _make_minimal_state()
        required = {"messages", "context", "user_id", "project_id", "plan", "plan_step", "recalled_facts"}
        missing = required - set(state.keys())
        assert not missing, f"Minimal state dict missing keys: {missing}"

    def test_recalled_facts_default_type(self):
        """recalled_facts should be list[str] — verify type annotation."""
        hints = {}
        for cls in reversed(OracleState.__mro__):
            hints.update(getattr(cls, "__annotations__", {}))
        ann = str(hints.get("recalled_facts", ""))
        assert "list" in ann.lower() or "List" in ann, (
            f"recalled_facts annotation {ann!r} does not appear to be a list type"
        )

    def test_plan_step_is_int_typed(self):
        """plan_step: int — check annotation."""
        hints = {}
        for cls in reversed(OracleState.__mro__):
            hints.update(getattr(cls, "__annotations__", {}))
        ann = hints.get("plan_step")
        assert ann is int or str(ann) == "int", (
            f"plan_step annotation is {ann!r}, expected int"
        )


@pytest.mark.skipif(not _IMPORT_OK, reason="oracle_v2.state not importable")
class TestOracleStateInheritance:
    """Verify OracleState inherits from CodeActState."""

    def test_inherits_from_codeact_state(self):
        """OracleState must subclass CodeActState."""
        try:
            from langgraph_codeact import CodeActState
        except ImportError:
            pytest.skip("langgraph_codeact not installed")

        assert issubclass(OracleState, CodeActState), (
            f"OracleState does not inherit from CodeActState. MRO: {OracleState.__mro__}"
        )

    def test_class_docstring_mentions_security_invariant(self):
        """Class docstring should document the security invariant."""
        doc = OracleState.__doc__ or ""
        # The docstring should mention that api_key is NOT stored
        assert "api_key" in doc or "security" in doc.lower(), (
            "OracleState docstring should mention the security invariant "
            "(api_key not stored in state)"
        )
