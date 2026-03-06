"""Pattern filter models for connector webhook triggers."""
from __future__ import annotations
import fnmatch
import re
from typing import Literal
from pydantic import BaseModel


class PatternCondition(BaseModel):
    field: str
    operator: Literal[
        "equals", "contains", "starts_with", "ends_with",
        "regex", "glob", "not_equals", "not_contains",
    ]
    value: str


class PatternFilter(BaseModel):
    """All conditions must match (AND logic)."""
    conditions: list[PatternCondition]


def match_condition(condition: PatternCondition, fields: dict[str, str]) -> bool:
    value = fields.get(condition.field, "")
    pattern = condition.value
    match condition.operator:
        case "equals":       return value.lower() == pattern.lower()
        case "contains":     return pattern.lower() in value.lower()
        case "starts_with":  return value.lower().startswith(pattern.lower())
        case "ends_with":    return value.lower().endswith(pattern.lower())
        case "glob":         return fnmatch.fnmatch(value.lower(), pattern.lower())
        case "regex":        return bool(re.search(pattern, value, re.IGNORECASE))
        case "not_equals":   return value.lower() != pattern.lower()
        case "not_contains": return pattern.lower() not in value.lower()
        case _:              return False


def match_pattern(pattern: PatternFilter, fields: dict[str, str]) -> bool:
    """Return True if all conditions in the filter match the given fields (AND logic)."""
    return all(match_condition(c, fields) for c in pattern.conditions)
