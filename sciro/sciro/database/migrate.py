"""
SCIRO Database Migration Script.

Initializes the database schema by executing schema.sql and views.sql.
Designed to be idempotent — safe to run multiple times.

Usage:
    python -m sciro.database.migrate
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def get_sql_file_path(filename: str) -> str:
    """Get the absolute path to a SQL file in the database directory.

    Args:
        filename: Name of the SQL file (e.g., 'schema.sql').

    Returns:
        Absolute path to the SQL file.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def run_migration() -> bool:
    """Execute the full database migration.

    Runs schema.sql (tables, constraints, indexes, triggers)
    followed by views.sql (analytics views).

    Returns:
        True if migration succeeded, False otherwise.
    """
    from sciro.config.logging_config import setup_logging
    setup_logging()

    from sciro.database.connection import execute_sql_file, check_connection, close_pool

    logger.info("=" * 60)
    logger.info("SCIRO Database Migration")
    logger.info("=" * 60)

    # Step 1: Check connection
    logger.info("Step 1: Checking database connection...")
    if not check_connection():
        logger.error(
            "Cannot connect to database. Please check your .env file "
            "and ensure PostgreSQL is running."
        )
        return False

    # Step 2: Execute schema.sql
    schema_path = get_sql_file_path("schema.sql")
    logger.info("Step 2: Creating tables and indexes...")
    try:
        execute_sql_file(schema_path)
        logger.info("Schema created successfully.")
    except FileNotFoundError:
        logger.error("schema.sql not found at: %s", schema_path)
        return False
    except Exception as e:
        logger.error("Schema creation failed: %s", e)
        return False

    # Step 3: Execute views.sql
    views_path = get_sql_file_path("views.sql")
    logger.info("Step 3: Creating analytics views...")
    try:
        execute_sql_file(views_path)
        logger.info("Views created successfully.")
    except FileNotFoundError:
        logger.error("views.sql not found at: %s", views_path)
        return False
    except Exception as e:
        logger.error("Views creation failed: %s", e)
        return False

    # Step 4: Verify tables exist
    logger.info("Step 4: Verifying migration...")
    try:
        from sciro.database.connection import get_cursor

        expected_tables = [
            "regions", "suppliers", "products", "inventory",
            "shipments", "customers", "orders", "order_items",
        ]

        with get_cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'sciro'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            existing_tables = [row[0] for row in cur.fetchall()]

        missing = set(expected_tables) - set(existing_tables)
        if missing:
            logger.error("Missing tables after migration: %s", missing)
            return False

        logger.info("All %d tables verified: %s", len(expected_tables), existing_tables)

        # Verify views
        with get_cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'sciro'
                ORDER BY table_name
            """)
            existing_views = [row[0] for row in cur.fetchall()]

        logger.info("Views created: %s", existing_views)

    except Exception as e:
        logger.warning("Verification query failed (non-critical): %s", e)

    logger.info("=" * 60)
    logger.info("Migration completed successfully!")
    logger.info("=" * 60)

    close_pool()
    return True


def main() -> None:
    """Entry point for migration script."""
    success = run_migration()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
