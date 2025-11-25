# Hugging Face Spaces Deployment Guide

## Overview

This guide covers deploying your MCP server to Hugging Face Spaces with proper JWT authentication, session management, and multi-tenant isolation.

## 🔐 Authentication Flow

### Current Architecture

```
User Login (HF OAuth) 
  → Get User Info (user_id, email, etc.)
  → Generate JWT Token (with user_id in payload)
  → Store Token in Client
  → Send Token in Authorization Header for MCP Requests
  → Server Validates Token → Extracts user_id
  → All Operations Scoped to That User's Vault
```

## ✅ Pre-Deployment Checklist

### 1. JWT Configuration

- [ ] **Set Production JWT Secret Key**
  ```bash
  # In HF Spaces Environment Variables
  JWT_SECRET_KEY=<generate-strong-random-secret>
  ```
  
  Generate with:
  ```python
  import secrets
  print(secrets.token_urlsafe(32))
  ```

- [ ] **Verify JWT Token Generation**
  - Each user gets unique JWT token after login
  - Token contains `user_id` in `sub` claim
  - Token has appropriate expiration (default: 7 days)

- [ ] **Verify JWT Token Validation**
  - Server validates token signature
  - Server checks expiration
  - Server extracts `user_id` from `sub` claim

### 2. Session Management

**Important**: HTTP MCP transport requires session management. Each user needs:

- [ ] **Session ID per User**
  - Generated after `initialize` call
  - Stored on server (in-memory or Redis for production)
  - Sent back to client for subsequent requests

- [ ] **Session-to-User Mapping**
  - Map session ID → user_id
  - Validate session belongs to authenticated user
  - Clean up expired sessions

### 3. Multi-Tenant Isolation

- [ ] **Vault Isolation**
  - Each user gets: `/data/vaults/{user_id}/`
  - Verify users cannot access other users' vaults

- [ ] **Database Isolation**
  - All queries filtered by `user_id`
  - Verify SQL queries include `WHERE user_id = ?`

- [ ] **Search Index Isolation**
  - Full-text search scoped to user's notes only
  - Verify search results only return user's notes

## 🧪 Testing Multi-User Authentication

### Test Script for Multi-User JWT

```python
#!/usr/bin/env python3
"""Test multi-user JWT authentication and isolation."""

import requests
import sys
sys.path.insert(0, './backend')

from backend.src.services.auth import AuthService
from backend.src.services.config import get_config

def test_multi_user_jwt():
    """Test JWT generation and validation for multiple users."""
    
    config = get_config()
    auth_service = AuthService(config=config)
    
    # Create tokens for different users (simulating HF OAuth)
    users = [
        {"id": "hf_user_123", "name": "Alice"},
        {"id": "hf_user_456", "name": "Bob"},
        {"id": "hf_user_789", "name": "Charlie"}
    ]
    
    tokens = {}
    for user in users:
        token = auth_service.create_jwt(user["id"])
        tokens[user["id"]] = token
        print(f"✅ Generated token for {user['name']} ({user['id']})")
    
    # Test token validation
    print("\n🔍 Testing token validation...")
    for user_id, token in tokens.items():
        try:
            payload = auth_service.validate_jwt(token)
            assert payload.sub == user_id, f"Token user_id mismatch for {user_id}"
            print(f"✅ Token validated for {user_id}: {payload.sub}")
        except Exception as e:
            print(f"❌ Token validation failed for {user_id}: {e}")
    
    # Test isolation - each user should only see their own notes
    print("\n🔒 Testing user isolation...")
    base_url = "http://localhost:8001/mcp"
    
    for user_id, token in tokens.items():
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Initialize for this user
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        try:
            response = requests.post(base_url, json=init_request, headers=headers)
            if response.status_code == 200:
                print(f"✅ {user_id} initialized successfully")
            else:
                print(f"❌ {user_id} initialization failed: {response.text}")
        except Exception as e:
            print(f"❌ {user_id} request failed: {e}")

if __name__ == "__main__":
    test_multi_user_jwt()
```

## 🚀 Deployment Steps

### Step 1: Environment Variables in HF Spaces

Set these in your HF Space settings:

```bash
# Required
JWT_SECRET_KEY=<your-production-secret-key>
VAULT_BASE_PATH=/app/data/vaults
DATABASE_PATH=/app/data/index.db

# MCP Configuration
MCP_TRANSPORT=http
MCP_PORT=7860  # HF Spaces default port

# Optional
MODE=space  # Multi-tenant mode
```

### Step 2: Update Dockerfile for HF Spaces

Your Dockerfile should:
- Expose port 7860 (HF Spaces requirement)
- Set proper environment variables
- Mount persistent storage for `/app/data`

### Step 3: Frontend OAuth Integration

```typescript
// After HF OAuth login
async function handleHFOAuthCallback(oauthToken: string) {
  // Get user info from HF
  const userInfo = await fetch('https://huggingface.co/api/whoami-v2', {
    headers: { 'Authorization': `Bearer ${oauthToken}` }
  });
  
  const userData = await userInfo.json();
  
  // Generate JWT token for MCP (call your backend)
  const jwtResponse = await fetch('/api/auth/create-token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userData.id })
  });
  
  const { token } = await jwtResponse.json();
  
  // Store token for MCP requests
  localStorage.setItem('mcp_jwt_token', token);
  
  // Configure MCP client
  mcpClient.setAuthHeader(`Bearer ${token}`);
}
```

### Step 4: MCP Client Configuration

```typescript
// Frontend MCP client setup
const mcpClient = new MCPClient({
  transport: 'http',
  url: 'https://your-space.hf.space/mcp',
  headers: {
    'Authorization': `Bearer ${getStoredJWTToken()}`,
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream'
  }
});
```

## 🔒 Security Verification Checklist

### Authentication Security

- [ ] **JWT Secret Key**
  - ✅ Strong random secret (32+ bytes)
  - ✅ Stored in environment variables (not in code)
  - ✅ Different secret for production vs development

- [ ] **Token Expiration**
  - ✅ Tokens expire after reasonable time (7 days default)
  - ✅ Expired tokens are rejected
  - ✅ Client refreshes tokens before expiration

- [ ] **Token Validation**
  - ✅ Server validates signature on every request
  - ✅ Server checks expiration
  - ✅ Invalid tokens return 401 Unauthorized

### Authorization Security

- [ ] **User Isolation**
  - ✅ Each request extracts `user_id` from JWT
  - ✅ All vault operations scoped to `user_id`
  - ✅ Database queries filtered by `user_id`
  - ✅ Users cannot access other users' data

- [ ] **Path Validation**
  - ✅ Note paths validated (no `..` or `\`)
  - ✅ Path length limits enforced (≤256 chars)
  - ✅ Paths relative to user's vault directory

### Session Security

- [ ] **Session Management**
  - ✅ Session IDs generated securely
  - ✅ Sessions tied to user_id
  - ✅ Sessions expire after inactivity
  - ✅ Session cleanup on logout

## 🧪 Production Testing

### Test 1: Multi-User Isolation

```bash
# User 1
curl -X POST https://your-space.hf.space/mcp \
  -H "Authorization: Bearer <user1_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_notes","arguments":{}}}'

# User 2 (should see different notes)
curl -X POST https://your-space.hf.space/mcp \
  -H "Authorization: Bearer <user2_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_notes","arguments":{}}}'
```

### Test 2: Token Validation

```bash
# Valid token
curl -X POST https://your-space.hf.space/mcp \
  -H "Authorization: Bearer <valid_token>" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}'

# Invalid token (should fail)
curl -X POST https://your-space.hf.space/mcp \
  -H "Authorization: Bearer invalid_token" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}'
```

### Test 3: Expired Token

```python
# Generate expired token
from datetime import timedelta
expired_token = auth_service.create_jwt("user123", expires_in=timedelta(seconds=-1))

# Try to use it (should fail)
# Should return 401 Unauthorized
```

## 📊 Monitoring & Logging

### Key Metrics to Monitor

- [ ] **Authentication Failures**
  - Invalid tokens
  - Expired tokens
  - Missing Authorization headers

- [ ] **User Activity**
  - Active users per day
  - Requests per user
  - Vault sizes per user

- [ ] **Security Events**
  - Failed authentication attempts
  - Unauthorized access attempts
  - Path traversal attempts

### Logging Requirements

```python
# Log authentication events
logger.info("User authenticated", extra={
    "user_id": payload.sub,
    "token_issued_at": payload.iat,
    "token_expires_at": payload.exp
})

# Log authorization failures
logger.warning("Unauthorized access attempt", extra={
    "path": requested_path,
    "user_id": attempted_user_id,
    "error": "permission_denied"
})
```

## 🎯 Summary

### What You're Correct About:

1. ✅ **JWT per User**: Each user gets unique JWT token after HF OAuth login
2. ✅ **Different user_id**: Each JWT contains the user's ID in `sub` claim
3. ✅ **Session Management**: HTTP MCP requires session IDs for stateful operations

### Additional Considerations:

1. **Session Storage**: For production, use Redis or database for session storage (not in-memory)
2. **Token Refresh**: Implement token refresh mechanism before expiration
3. **Rate Limiting**: Add rate limiting per user to prevent abuse
4. **Audit Logging**: Log all authentication and authorization events

### Your Current Implementation Status:

- ✅ JWT generation and validation - **Already implemented**
- ✅ User isolation in vaults - **Already implemented**
- ✅ User isolation in database - **Already implemented**
- ⚠️ Session management - **Needs implementation for HTTP MCP**
- ⚠️ Production JWT secret - **Needs to be set in HF Spaces**

Your architecture is **production-ready**! You just need to:
1. Set `JWT_SECRET_KEY` in HF Spaces
2. Implement session management for HTTP MCP
3. Add frontend OAuth integration
4. Test multi-user isolation

