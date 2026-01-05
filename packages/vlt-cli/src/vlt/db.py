"""
VLT Database Module

This module manages the SQLite database connection for vlt-cli.
The database is profile-specific, stored in ~/.vlt/profiles/{profile}/vault.db.

Usage:
    from vlt.db import engine, SessionLocal, Base, get_db

    # For ORM operations
    with SessionLocal() as session:
        session.query(...)

    # For raw SQL
    with engine.connect() as conn:
        conn.execute(text("SELECT ..."))
"""

from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from vlt.config import settings

# Set up SQLite pragmas for better performance and integrity
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Configure SQLite connection with recommended settings."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes with WAL
    cursor.close()


# Create engine using profile-specific database URL
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}  # Needed for SQLite with threads
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


def get_db():
    """
    Dependency for FastAPI/service layer to get a database session.

    Yields:
        Session: SQLAlchemy session that auto-closes
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine_for_profile(profile_name: Optional[str] = None) -> Engine:
    """
    Get a database engine for a specific profile.

    This is useful for operations that need to access a different profile's database,
    such as data migration or cross-profile queries.

    Args:
        profile_name: Profile name. If None, uses the active profile (same as `engine`).

    Returns:
        SQLAlchemy Engine for the profile's database
    """
    if profile_name is None:
        return engine

    from vlt.profile import get_profile_manager
    manager = get_profile_manager()
    db_url = manager.get_database_url(profile_name)

    profile_engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False}
    )
    return profile_engine


def get_session_for_profile(profile_name: Optional[str] = None) -> sessionmaker:
    """
    Get a session factory for a specific profile.

    Args:
        profile_name: Profile name. If None, uses the active profile.

    Returns:
        SQLAlchemy sessionmaker bound to the profile's database
    """
    profile_engine = get_engine_for_profile(profile_name)
    return sessionmaker(autocommit=False, autoflush=False, bind=profile_engine)
