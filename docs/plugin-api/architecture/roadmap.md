# Roadmap

This document outlines stretch goals and future enhancements for the Oracle Plugin System.

## Current State (v1.0)

- TOML rule definitions with simpleeval expressions
- Lua script escape hatch via lupa (LuaJIT)
- Four built-in rules (token budget, iteration budget, large result, repeated failure)
- SQLite-backed plugin state persistence
- Settings UI for rule management
- **Behavior Tree Architecture** (Honorbuddy-style)
  - PrioritySelector, Sequence, Parallel composites
  - Guard, Inverter, Cooldown decorators
  - Frame locking optimization for O(1) resume
  - Context detection for environment-aware behavior

## Near-Term Enhancements

### Cython Behavior Tree Core

**Priority**: High
**Timeline**: v1.1

Migrate behavior tree hot path to Cython for performance:

```
backend/src/services/plugins/behavior_tree/
+-- types.pyx             # RunStatus, TickContext, Blackboard
+-- node.pyx              # BehaviorNode base class
+-- composites.pyx        # PrioritySelector, Sequence, Parallel
+-- decorators.pyx        # Inverter, Guard, Cooldown
+-- leaves.pyx            # ConditionNode, ActionNode, ScriptNode
+-- tree.pyx              # BehaviorTree with frame locking
```

**Performance targets**:
| Metric | Current (Python) | Target (Cython) |
|--------|-----------------|-----------------|
| Single node tick | 1-10 ms | <100 ns |
| Tree traversal (100 nodes) | 10-50 ms | <10 us |
| Condition evaluation | 0.1-1 ms | <1 us |
| Full hook point cycle | 1-5 ms | <100 us |

**Build setup**:
```python
# backend/src/services/plugins/setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "behavior_tree/types.pyx",
        "behavior_tree/node.pyx",
        "behavior_tree/composites.pyx",
        "behavior_tree/decorators.pyx",
        "behavior_tree/leaves.pyx",
        "behavior_tree/tree.pyx",
    ], language_level="3"),
)
```

### Expression Language Extensions

**Priority**: Medium
**Timeline**: v1.1

Add more built-in functions for common patterns:

```toml
# Proposed new functions
expression = "recent_tool_count('vault_search', 5) > 3"
expression = "time_since_last_tool('search') > 60"
expression = "session_duration_minutes() > 30"
```

### Rule Templates

**Priority**: Medium
**Timeline**: v1.1

Pre-built rule templates for common scenarios:

```bash
vlt plugin create --template research-workflow
vlt plugin create --template code-review
vlt plugin create --template documentation
```

### Plugin Marketplace

**Priority**: Low
**Timeline**: v1.2

Community-contributed plugins with:
- Discovery API
- Version management
- Dependency resolution
- Security review process

## Long-Term Vision

### BERT Semantic Conditions

**Priority**: Low
**Timeline**: v2.0

Semantic pattern matching for natural language conditions:

```toml
[condition]
semantic = { concept = "authentication", field = "tool.args.query", threshold = 0.7 }
```

**Implementation approach**:
1. Pre-compute concept embeddings (auth, security, performance, etc.)
2. Embed incoming text at runtime using `sentence-transformers/all-MiniLM-L6-v2`
3. Compare via cosine similarity
4. Cache embeddings per session

**Use cases**:
- Detect when agent is working on security-sensitive code
- Identify performance optimization patterns
- Recognize documentation tasks

### LISP Rule Language

**Priority**: Low
**Timeline**: v2.0

S-expression alternative for power users:

```lisp
(and (> (ctx-token-usage) 0.8)
     (any (ctx-recent-tools 5)
          (lambda (t) (eq (tool-name t) "vault_search"))))
```

**Implementation options**:
- Use `hy` (Python-hosted Lisp)
- Custom S-expression parser
- Compile to simpleeval expressions

### Skills/Behaviors System

**Priority**: Medium
**Timeline**: v2.0

Higher-level plugins with complex multi-turn logic:

```toml
[skill]
id = "deep-researcher"
name = "Deep Research Workflow"
type = "behavior"

[workflow]
stages = ["gather", "analyze", "synthesize", "report"]
max_iterations = 50
checkpoint_interval = 10
```

**Proposed skills**:
- **Deep Researcher**: Multi-step research with source synthesis
- **Agent Swarm**: Coordinate multiple sub-agents
- **Code Reviewer**: Automated code review workflow
- **Documentation Writer**: Generate docs from code analysis

**Requirements**:
- State machine support
- Multi-turn workflow persistence
- Agent communication protocols
- Progress tracking UI

### Rust Performance Core

**Priority**: Low
**Timeline**: v2.0+

When performance requirements exceed 10k rules/sec:

```rust
// rust-rule-engine crate
#[pyfunction]
fn evaluate_rules(rules: Vec<Rule>, context: Context) -> Vec<RuleResult> {
    // Rete-style pattern matching
}
```

**Migration path**:
1. Create Rust crate with `maturin init --bindings pyo3`
2. Implement Rete-style pattern matching in Rust
3. Expose as Python module via `#[pyfunction]` decorators
4. Python continues orchestration; Rust handles hot path

**Expected gains**: 10-100x performance improvement

### Visual Rule Builder

**Priority**: Low
**Timeline**: v2.0+

Web-based UI for creating rules without writing TOML:

- Drag-and-drop condition builder
- Action configuration wizard
- Live preview and testing
- Import/export as TOML

## Community Contributions

Areas open for community contribution:

1. **New built-in rules**: Submit PRs for common patterns
2. **Expression functions**: Add helper functions
3. **Lua libraries**: Safe utility libraries for scripts
4. **Documentation**: Examples and tutorials
5. **Testing**: Integration test coverage

## Deprecation Policy

- Major version changes may deprecate features
- Deprecated features supported for 2 minor versions
- Migration guides provided for breaking changes
- Core rules never deprecated (may evolve)

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-01 | Initial release |

## Feedback

Feature requests and feedback welcome:
- GitHub Issues for bugs and feature requests
- Discussions for design proposals
- PRs for contributions

## See Also

- [Architecture Overview](./overview.md)
- [Decision Tree Architecture](./decision-tree.md)
- [Performance Considerations](./performance.md)
- [Advanced Patterns](../rules/advanced-patterns.md)
- [Behavior Tree Tasks](../../../specs/015-oracle-plugin-system/behavior-tree-tasks.md)
- [Research Notes](../../../specs/015-oracle-plugin-system/research.md)
