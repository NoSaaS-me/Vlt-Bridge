# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2026-01-01 13:29]
ENABLE_NOAUTH_MCP bypasses ALL authentication in the API - it affects both HTTP routes (via get_auth_context) and MCP HTTP transport (via _current_user_id). Any routes using get_auth_context will accept unauthenticated requests when this flag is true.

_Context: The bypass was added for hackathon/demo purposes but creates a significant security vulnerability if enabled in production. The demo-user check _ensure_write_allowed only prevents writes, not reads._
