"""
SCIRO Database Connection Manager.

Provides a connection pool and context managers for safe
PostgreSQL connection/cursor handling with automatic
transaction commit/rollback.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator, Any

import psycopg2
import psycopg2.pool
import psycopg2.extras

from sciro.config.settings import get_settings

logger = logging.getLogger(__name__)

# Module-level connection pool (initialized lazily)
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def get_pool(min_conn: int = 1, max_conn: int = 10) -> psycopg2.pool.ThreadedConnectionPool:
    """Get or create the database connection pool.

    Args:
        min_conn: Minimum number of connections to maintain.
        max_conn: Maximum number of connections in the pool.

    Returns:
        ThreadedConnectionPool instance.

    Raises:
        psycopg2.OperationalError: If database connection fails.
    """
    global _pool

    if _pool is None or _pool.closed:
        settings = get_settings()
        logger.info(
            "Creating connection pool: %s@%s:%s/%s",
            settings.db.user, settings.db.host, settings.db.port, settings.db.name,
        )
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                **settings.db.connection_dict,
            )
            logger.info("Connection pool created successfully.")
        except psycopg2.OperationalError as e:
            logger.error("Failed to create connection pool: %s", e)
            raise

    return _pool


def close_pool() -> None:
    """Close all connections in the pool."""
    global _pool
    if _pool is not None and not _pool.closed:
        _pool.closeall()
        logger.info("Connection pool closed.")
        _pool = None


@contextmanager
def get_connection() -> Generator:
    """Context manager that provides a database connection from the pool.

    The connection is automatically returned to the pool on exit.
    Transactions are committed on success, rolled back on exception.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    Yields:
        psycopg2 connection object.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(
    cursor_factory: Any = None,
) -> Generator:
    """Context manager that provides a database cursor.

    Handles connection acquisition, transaction commit/rollback,
    and connection return to pool automatically.

    Args:
        cursor_factory: Optional cursor factory (e.g., psycopg2.extras.DictCursor).

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM regions")
            rows = cur.fetchall()

        with get_cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT * FROM regions")
            rows = cur.fetchall()  # Returns dicts

    Yields:
        psycopg2 cursor object.
    """
    with get_connection() as conn:
        kwargs = {}
        if cursor_factory is not None:
            kwargs["cursor_factory"] = cursor_factory
        cur = conn.cursor(**kwargs)
        try:
            yield cur
        finally:
            cur.close()


def check_connection() -> bool:
    """Verify database connectivity.

    Returns:
        True if connection is successful, False otherwise.
    """
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            if result and result[0] == 1:
                logger.info("Database connection check: OK")
                return True
    except Exception as e:
        logger.error("Database connection check failed: %s", e)
    return False


def execute_sql_file(filepath: str) -> None:
    """Execute a SQL file against the database.

    Args:
        filepath: Path to the SQL file to execute.

    Raises:
        FileNotFoundError: If the SQL file doesn't exist.
        psycopg2.Error: If SQL execution fails.
    """
    import os

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SQL file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        sql_content = f.read()

    logger.info("Executing SQL file: %s", filepath)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_content)

    logger.info("SQL file executed successfully: %s", filepath)
