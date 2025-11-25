#!/usr/bin/env python3
"""Generate a JWT token for testing MCP HTTP transport."""

import sys
import os

# Add backend to path
sys.path.insert(0, "./backend")

from backend.src.services.auth import AuthService
from backend.src.services.config import get_config


def generate_token(user_id="local-dev"):
    """Generate a JWT token for the specified user."""
    try:
        config = get_config()
        auth_service = AuthService(config=config)

        # Generate JWT token
        token = auth_service.create_jwt(user_id)

        print(f"✅ Generated JWT token for user '{user_id}':")
        print(f"Bearer {token}")
        print(f"\n📋 Copy this to your mcp.json:")
        print(f'"Authorization": "Bearer {token}"')

        return token

    except Exception as e:
        print(f"❌ Error generating token: {e}")
        print("💡 Make sure JWT_SECRET_KEY is set in your environment")
        return None


if __name__ == "__main__":
    user_id = sys.argv[1] if len(sys.argv) > 1 else "local-dev"
    generate_token(user_id)
