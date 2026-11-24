"""
SCIRO Configuration Settings.

Loads environment variables from .env file and provides
typed access to all configuration parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv


# Determine project root (sciro/ directory containing .env)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env() -> None:
    """Load .env file from project root."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try .env.example as fallback for development
        example_path = _PROJECT_ROOT / ".env.example"
        if example_path.exists():
            load_dotenv(example_path)


# Load environment on module import
_load_env()


@dataclass(frozen=True)
class DatabaseSettings:
    """PostgreSQL connection settings."""

    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "sciro_db"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    @property
    def connection_string(self) -> str:
        """Return psycopg2-compatible connection string."""
        return (
            f"host={self.host} port={self.port} dbname={self.name} "
            f"user={self.user} password={self.password}"
        )

    @property
    def connection_dict(self) -> dict:
        """Return connection parameters as a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.name,
            "user": self.user,
            "password": self.password,
        }


@dataclass(frozen=True)
class Settings:
    """Application-wide settings."""

    # Database
    db: DatabaseSettings = field(default_factory=DatabaseSettings)

    # Paths
    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_DIR: Path = field(
        default_factory=lambda: _PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    )

    # Logging
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def SAMPLE_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "sample"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
