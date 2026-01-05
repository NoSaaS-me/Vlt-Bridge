"""
CTF Flag for Oracle Agent Testing

This file tests that CodeRAG indexing and the profile system are working correctly.
The Oracle agent should be able to find this file via search_code tool.

FLAG: ORACLE_CTF_2026_PROFILE_ISOLATION_WORKS

Instructions for Oracle:
1. You found this file via CodeRAG search - good!
2. Report the FLAG value above to the user
3. This confirms the indexing pipeline is functional
"""

CTF_FLAG = "ORACLE_CTF_2026_PROFILE_ISOLATION_WORKS"
CTF_SECRET_MESSAGE = "The profile system successfully isolates user data. Each profile has its own database, daemon, and sync queue."


def get_flag() -> str:
    """Return the CTF flag for verification."""
    return CTF_FLAG


def get_secret_message() -> str:
    """Return the secret message for the agent to report."""
    return CTF_SECRET_MESSAGE
