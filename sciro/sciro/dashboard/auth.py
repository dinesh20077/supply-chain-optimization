"""
SCIRO Authentication & Role-Based Access Control.

Simple but complete auth system for the Streamlit dashboard:
- Password hashing with bcrypt (fallback to hashlib)
- JWT-like session tokens
- Role-based access (admin, analyst, viewer)
- Session management via Streamlit session_state
"""

import logging
import hashlib
import hmac
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User model."""
    username: str
    password_hash: str
    role: str  # 'admin', 'analyst', 'viewer'
    full_name: str = ""
    email: str = ""
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# Role permissions matrix
ROLE_PERMISSIONS = {
    "admin": {
        "view_dashboard", "view_analytics", "view_supply_chain",
        "view_inventory", "view_revenue", "view_customers",
        "run_simulations", "run_ml_models", "optimize_pricing",
        "manage_users", "export_data", "configure_settings",
    },
    "analyst": {
        "view_dashboard", "view_analytics", "view_supply_chain",
        "view_inventory", "view_revenue", "view_customers",
        "run_simulations", "run_ml_models", "optimize_pricing",
        "export_data",
    },
    "viewer": {
        "view_dashboard", "view_analytics", "view_supply_chain",
        "view_inventory", "view_revenue", "view_customers",
    },
}


class AuthManager:
    """Authentication and authorization manager."""

    # Secret key for token signing (generated at startup)
    _secret_key: str = secrets.token_hex(32)

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._setup_default_users()

    def _setup_default_users(self):
        """Create default users for demo purposes."""
        defaults = [
            ("admin", "admin123", "admin", "System Admin", "admin@sciro.ai"),
            ("analyst", "analyst123", "analyst", "Data Analyst", "analyst@sciro.ai"),
            ("viewer", "viewer123", "viewer", "Dashboard Viewer", "viewer@sciro.ai"),
        ]
        for username, password, role, name, email in defaults:
            self.create_user(username, password, role, name, email)

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        hash_value = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"{salt}:{hash_value}"

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, hash_value = stored_hash.split(":")
            check_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return hmac.compare_digest(hash_value, check_hash)
        except (ValueError, AttributeError):
            return False

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "viewer",
        full_name: str = "",
        email: str = "",
    ) -> bool:
        """Create a new user."""
        if username in self._users:
            logger.warning("User '%s' already exists", username)
            return False

        if role not in ROLE_PERMISSIONS:
            logger.error("Invalid role: %s", role)
            return False

        self._users[username] = User(
            username=username,
            password_hash=self._hash_password(password),
            role=role,
            full_name=full_name,
            email=email,
        )
        logger.info("Created user '%s' with role '%s'", username, role)
        return True

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a session token.

        Returns:
            Session token string if successful, None if failed.
        """
        user = self._users.get(username)
        if not user:
            logger.warning("Login failed: user '%s' not found", username)
            return None

        if not user.is_active:
            logger.warning("Login failed: user '%s' is inactive", username)
            return None

        if not self._verify_password(password, user.password_hash):
            logger.warning("Login failed: wrong password for '%s'", username)
            return None

        # Create session
        token = secrets.token_urlsafe(32)
        self._sessions[token] = {
            "username": username,
            "role": user.role,
            "full_name": user.full_name,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=8)).isoformat(),
        }

        logger.info("User '%s' authenticated successfully", username)
        return token

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a session token.

        Returns:
            Session data dict if valid, None if invalid/expired.
        """
        session = self._sessions.get(token)
        if not session:
            return None

        expires = datetime.fromisoformat(session["expires_at"])
        if datetime.now() > expires:
            del self._sessions[token]
            return None

        return session

    def check_permission(self, token: str, permission: str) -> bool:
        """Check if the user with this token has a specific permission."""
        session = self.validate_session(token)
        if not session:
            return False
        role = session["role"]
        return permission in ROLE_PERMISSIONS.get(role, set())

    def logout(self, token: str) -> None:
        """Invalidate a session."""
        if token in self._sessions:
            del self._sessions[token]

    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (without password hashes)."""
        return [
            {
                "username": u.username,
                "role": u.role,
                "full_name": u.full_name,
                "email": u.email,
                "is_active": u.is_active,
                "created_at": u.created_at,
            }
            for u in self._users.values()
        ]

    def update_role(self, username: str, new_role: str) -> bool:
        """Update a user's role."""
        if username not in self._users:
            return False
        if new_role not in ROLE_PERMISSIONS:
            return False
        self._users[username].role = new_role
        logger.info("Updated role for '%s' to '%s'", username, new_role)
        return True

    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account."""
        if username not in self._users:
            return False
        self._users[username].is_active = False
        # Invalidate their sessions
        to_remove = [
            token for token, session in self._sessions.items()
            if session["username"] == username
        ]
        for token in to_remove:
            del self._sessions[token]
        logger.info("Deactivated user '%s'", username)
        return True

    def get_role_permissions(self, role: str) -> set:
        """Get permissions for a role."""
        return ROLE_PERMISSIONS.get(role, set())
