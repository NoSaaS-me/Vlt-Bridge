# Enforce authentication on sensitive API routes

## Overview

Most API routes do not enforce authentication despite handling sensitive operations. The project index shows 64 routes with 'requires_auth: false'. Critical endpoints like /api/notes (CRUD), /api/index/rebuild, /oracle/stream, /threads/*, and /projects/* are accessible without authentication when ENABLE_NOAUTH_MCP is true or via the demo-user bypass.

## Rationale

Broken access control is #1 on OWASP Top 10 (2021). Unauthenticated access to user data and administrative functions allows data theft, data corruption, and denial of service.

---
*This spec was created from ideation and is pending detailed specification.*
