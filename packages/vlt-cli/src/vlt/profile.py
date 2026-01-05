"""
VLT Profile Manager - Named profile system for multi-user isolation.

This module provides complete isolation between different vlt users/contexts via named profiles.
Each profile has its own:
- SQLite database (vault.db)
- Environment configuration (.env with sync token, etc.)
- Daemon PID file (daemon.pid)
- Log file (daemon.log)

Directory structure:
    ~/.vlt/
    ├── config.toml              # Active profile pointer + global settings
    └── profiles/
        ├── default/             # Default profile (migrated from legacy structure)
        │   ├── vault.db
        │   ├── .env
        │   ├── daemon.pid
        │   └── daemon.log
        └── {profile-name}/      # Additional named profiles
            ├── vault.db
            ├── .env
            ├── daemon.pid
            └── daemon.log

Usage:
    manager = ProfileManager()
    profile_name = manager.get_active_profile()  # Returns "default" or selected profile
    profile_dir = manager.get_profile_dir()       # Returns Path to profile directory
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

import tomli_w

logger = logging.getLogger(__name__)


# Valid profile name pattern: lowercase alphanumeric, hyphens, underscores
PROFILE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MAX_PROFILE_NAME_LENGTH = 64


class ProfileError(Exception):
    """Exception raised for profile-related errors."""
    pass


class ProfileManager:
    """
    Manages vlt named profiles with isolated storage.

    Provides:
    - Profile creation, deletion, and listing
    - Active profile switching
    - Profile directory resolution
    - Automatic migration from legacy structure
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the profile manager.

        Args:
            base_dir: Base vlt directory. Defaults to ~/.vlt
        """
        self.base_dir = base_dir or Path.home() / ".vlt"
        self.config_file = self.base_dir / "config.toml"
        self.profiles_dir = self.base_dir / "profiles"
        self._cached_active_profile: Optional[str] = None

    def _ensure_base_dir(self) -> None:
        """Ensure the base vlt directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_profiles_dir(self) -> None:
        """Ensure the profiles directory exists."""
        self._ensure_base_dir()
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _validate_profile_name(self, name: str) -> None:
        """
        Validate a profile name.

        Args:
            name: Profile name to validate

        Raises:
            ProfileError: If the name is invalid
        """
        if not name:
            raise ProfileError("Profile name cannot be empty")

        if len(name) > MAX_PROFILE_NAME_LENGTH:
            raise ProfileError(
                f"Profile name too long (max {MAX_PROFILE_NAME_LENGTH} characters)"
            )

        if not PROFILE_NAME_PATTERN.match(name):
            raise ProfileError(
                f"Invalid profile name '{name}'. "
                "Must start with lowercase letter or digit, "
                "and contain only lowercase letters, digits, hyphens, and underscores."
            )

        # Reserved names
        reserved = {"profiles", "config", "global"}
        if name in reserved:
            raise ProfileError(f"Profile name '{name}' is reserved")

    def _read_config(self) -> dict:
        """
        Read the config.toml file.

        Returns:
            Config dictionary, or empty dict if file doesn't exist.
        """
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning(f"Error reading config.toml: {e}")
            return {}

    def _write_config(self, config: dict) -> None:
        """
        Write the config.toml file.

        Args:
            config: Configuration dictionary to write
        """
        self._ensure_base_dir()

        try:
            with open(self.config_file, "wb") as f:
                tomli_w.dump(config, f)
        except Exception as e:
            raise ProfileError(f"Failed to write config.toml: {e}")

    def needs_migration(self) -> bool:
        """
        Check if legacy data needs migration to profile structure.

        Migration is needed if:
        - ~/.vlt/vault.db exists (legacy single-user location)
        - ~/.vlt/profiles/ does not exist

        Returns:
            True if migration is needed
        """
        legacy_db = self.base_dir / "vault.db"
        return legacy_db.exists() and not self.profiles_dir.exists()

    def migrate_legacy_data(self) -> None:
        """
        Migrate legacy data to the default profile.

        Moves:
        - ~/.vlt/vault.db -> ~/.vlt/profiles/default/vault.db
        - ~/.vlt/.env -> ~/.vlt/profiles/default/.env
        - ~/.vlt/daemon.pid -> ~/.vlt/profiles/default/daemon.pid
        - ~/.vlt/daemon.log -> ~/.vlt/profiles/default/daemon.log

        Creates config.toml with active_profile = "default"
        """
        if not self.needs_migration():
            logger.debug("No legacy data to migrate")
            return

        logger.info("Migrating legacy vlt data to default profile...")

        # Create default profile directory
        default_profile = self.profiles_dir / "default"
        default_profile.mkdir(parents=True, exist_ok=True)

        # Files to migrate
        migrate_files = [
            ("vault.db", "vault.db"),
            (".env", ".env"),
            ("daemon.pid", "daemon.pid"),
            ("daemon.log", "daemon.log"),
        ]

        for legacy_name, profile_name in migrate_files:
            legacy_path = self.base_dir / legacy_name
            profile_path = default_profile / profile_name

            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(profile_path))
                    logger.info(f"  Migrated {legacy_name} -> profiles/default/{profile_name}")
                except Exception as e:
                    logger.warning(f"  Failed to migrate {legacy_name}: {e}")

        # Create config.toml
        self._write_config({"active_profile": "default"})
        logger.info("Migration complete. Active profile: default")

    def get_active_profile(self) -> str:
        """
        Get the currently active profile name.

        Resolution order:
        1. VLT_PROFILE environment variable (overrides all)
        2. Project-level .vlt/profile file (in CWD or git root)
        3. config.toml active_profile setting
        4. Default to "default"

        Returns:
            Active profile name
        """
        # Check cache first
        if self._cached_active_profile is not None:
            return self._cached_active_profile

        # 1. Environment variable override
        env_profile = os.environ.get("VLT_PROFILE")
        if env_profile:
            try:
                self._validate_profile_name(env_profile)
                self._cached_active_profile = env_profile
                return env_profile
            except ProfileError:
                logger.warning(f"Invalid VLT_PROFILE value: {env_profile}, ignoring")

        # 2. Project-level override (.vlt/profile file)
        project_profile = self._get_project_profile()
        if project_profile:
            self._cached_active_profile = project_profile
            return project_profile

        # 3. Global config file
        config = self._read_config()
        active = config.get("active_profile", "default")

        # Validate
        try:
            self._validate_profile_name(active)
        except ProfileError:
            logger.warning(f"Invalid active_profile in config: {active}, using 'default'")
            active = "default"

        self._cached_active_profile = active
        return active

    def _get_project_profile(self) -> Optional[str]:
        """
        Get profile override from project-level .vlt/profile file.

        Searches in current directory, then walks up to git root.

        Returns:
            Profile name from project file, or None if not found
        """
        # Try current directory first
        cwd = Path.cwd()
        profile_file = cwd / ".vlt" / "profile"

        if profile_file.exists():
            return self._read_project_profile(profile_file)

        # Walk up to find git root
        current = cwd
        while current != current.parent:
            git_dir = current / ".git"
            if git_dir.exists():
                profile_file = current / ".vlt" / "profile"
                if profile_file.exists():
                    return self._read_project_profile(profile_file)
                break  # Stop at git root even if no profile file
            current = current.parent

        return None

    def _read_project_profile(self, profile_file: Path) -> Optional[str]:
        """
        Read a project-level profile file.

        Args:
            profile_file: Path to .vlt/profile file

        Returns:
            Profile name, or None if invalid
        """
        try:
            content = profile_file.read_text().strip()
            if content:
                self._validate_profile_name(content)
                return content
        except ProfileError as e:
            logger.warning(f"Invalid project profile file {profile_file}: {e}")
        except Exception as e:
            logger.warning(f"Error reading project profile {profile_file}: {e}")

        return None

    def set_active_profile(self, name: str) -> None:
        """
        Set the active profile in config.toml.

        Args:
            name: Profile name to activate

        Raises:
            ProfileError: If the profile doesn't exist or name is invalid
        """
        self._validate_profile_name(name)

        # Verify profile exists
        profile_dir = self.profiles_dir / name
        if not profile_dir.exists():
            raise ProfileError(f"Profile '{name}' does not exist. Create it first with 'vlt profile add {name}'")

        # Update config
        config = self._read_config()
        config["active_profile"] = name
        self._write_config(config)

        # Clear cache
        self._cached_active_profile = None

        logger.info(f"Active profile set to: {name}")

    def get_profile_dir(self, name: Optional[str] = None) -> Path:
        """
        Get the directory path for a profile.

        Args:
            name: Profile name. If None, uses active profile.

        Returns:
            Path to the profile directory

        Raises:
            ProfileError: If profile name is invalid
        """
        profile_name = name or self.get_active_profile()
        self._validate_profile_name(profile_name)

        profile_dir = self.profiles_dir / profile_name

        # Ensure directory exists if it's the default profile
        if profile_name == "default" and not profile_dir.exists():
            profile_dir.mkdir(parents=True, exist_ok=True)

        return profile_dir

    def list_profiles(self) -> list[str]:
        """
        List all available profiles.

        Returns:
            List of profile names, sorted alphabetically.
            Always includes "default" even if no profiles exist yet.
        """
        profiles = set()

        # Add default (always available)
        profiles.add("default")

        # Scan profiles directory
        if self.profiles_dir.exists():
            for item in self.profiles_dir.iterdir():
                if item.is_dir():
                    try:
                        self._validate_profile_name(item.name)
                        profiles.add(item.name)
                    except ProfileError:
                        # Skip invalid profile directories
                        pass

        return sorted(profiles)

    def create_profile(
        self,
        name: str,
        token: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> Path:
        """
        Create a new profile.

        Args:
            name: Profile name
            token: Optional sync token to store in profile's .env
            server_url: Optional server URL to store in profile's .env

        Returns:
            Path to the created profile directory

        Raises:
            ProfileError: If profile already exists or name is invalid
        """
        self._validate_profile_name(name)

        profile_dir = self.profiles_dir / name
        if profile_dir.exists():
            raise ProfileError(f"Profile '{name}' already exists")

        # Create profile directory
        self._ensure_profiles_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Create .env if token provided
        if token or server_url:
            env_path = profile_dir / ".env"
            lines = []
            if token:
                lines.append(f"VLT_SYNC_TOKEN={token}\n")
            if server_url:
                lines.append(f"VLT_VAULT_URL={server_url}\n")
            env_path.write_text("".join(lines))

        logger.info(f"Created profile: {name}")
        return profile_dir

    def delete_profile(self, name: str, force: bool = False) -> None:
        """
        Delete a profile and all its data.

        Args:
            name: Profile name to delete
            force: Skip confirmation (for programmatic use)

        Raises:
            ProfileError: If profile doesn't exist, is active, or is "default"
        """
        self._validate_profile_name(name)

        if name == "default":
            raise ProfileError("Cannot delete the default profile")

        profile_dir = self.profiles_dir / name
        if not profile_dir.exists():
            raise ProfileError(f"Profile '{name}' does not exist")

        # Check if active
        if self.get_active_profile() == name:
            raise ProfileError(
                f"Cannot delete active profile '{name}'. "
                "Switch to another profile first with 'vlt profile use <other>'"
            )

        # Delete the profile directory
        try:
            shutil.rmtree(profile_dir)
            logger.info(f"Deleted profile: {name}")
        except Exception as e:
            raise ProfileError(f"Failed to delete profile '{name}': {e}")

    def profile_exists(self, name: str) -> bool:
        """
        Check if a profile exists.

        Args:
            name: Profile name to check

        Returns:
            True if the profile exists
        """
        try:
            self._validate_profile_name(name)
            profile_dir = self.profiles_dir / name
            return profile_dir.exists()
        except ProfileError:
            return False

    def get_database_url(self, profile_name: Optional[str] = None) -> str:
        """
        Get the SQLite database URL for a profile.

        Args:
            profile_name: Profile name. If None, uses active profile.

        Returns:
            SQLite database URL (e.g., "sqlite:///~/.vlt/profiles/default/vault.db")
        """
        profile_dir = self.get_profile_dir(profile_name)
        db_path = profile_dir / "vault.db"
        return f"sqlite:///{db_path}"

    def get_env_file(self, profile_name: Optional[str] = None) -> Path:
        """
        Get the .env file path for a profile.

        Args:
            profile_name: Profile name. If None, uses active profile.

        Returns:
            Path to the profile's .env file
        """
        profile_dir = self.get_profile_dir(profile_name)
        return profile_dir / ".env"

    def get_pid_file(self, profile_name: Optional[str] = None) -> Path:
        """
        Get the daemon PID file path for a profile.

        Args:
            profile_name: Profile name. If None, uses active profile.

        Returns:
            Path to the profile's daemon.pid file
        """
        profile_dir = self.get_profile_dir(profile_name)
        return profile_dir / "daemon.pid"

    def get_log_file(self, profile_name: Optional[str] = None) -> Path:
        """
        Get the daemon log file path for a profile.

        Args:
            profile_name: Profile name. If None, uses active profile.

        Returns:
            Path to the profile's daemon.log file
        """
        profile_dir = self.get_profile_dir(profile_name)
        return profile_dir / "daemon.log"

    def get_daemon_port(self, profile_name: Optional[str] = None) -> int:
        """
        Get a unique daemon port for a profile.

        Uses a deterministic hash to assign ports in range 8765-9764.
        This allows multiple profiles to run daemons simultaneously.

        Args:
            profile_name: Profile name. If None, uses active profile.

        Returns:
            Port number for the daemon
        """
        name = profile_name or self.get_active_profile()

        # Default profile uses base port
        if name == "default":
            return 8765

        # Hash profile name to get port offset (0-999)
        hash_val = hash(name) % 1000
        return 8765 + hash_val

    def clear_cache(self) -> None:
        """Clear the cached active profile. Call when profile switching."""
        self._cached_active_profile = None


# Global singleton instance
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager(base_dir: Optional[Path] = None) -> ProfileManager:
    """
    Get the global ProfileManager instance.

    Args:
        base_dir: Override base directory (for testing)

    Returns:
        ProfileManager singleton
    """
    global _profile_manager

    if _profile_manager is None or base_dir is not None:
        _profile_manager = ProfileManager(base_dir)

    return _profile_manager


def get_active_profile_dir() -> Path:
    """
    Get the directory for the currently active profile.

    Convenience function for common use case.

    Returns:
        Path to active profile directory
    """
    return get_profile_manager().get_profile_dir()


def ensure_profile_migrated() -> None:
    """
    Ensure legacy data is migrated to profile structure.

    Call this early in CLI startup to handle migration transparently.
    """
    manager = get_profile_manager()
    if manager.needs_migration():
        manager.migrate_legacy_data()
