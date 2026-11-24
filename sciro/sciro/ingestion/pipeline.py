"""
SCIRO Data Ingestion Pipeline.

Orchestrates the full data ingestion flow:
    Load file → Validate → Clean → Transform → Insert into PostgreSQL

Supports loading all entity types from CSV/Excel files into the database.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from sciro.ingestion.loader import load_file
from sciro.ingestion.validators import validate_dataframe
from sciro.etl.cleaning import clean_dataframe
from sciro.config.settings import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Ingestion Report
# =============================================================================
@dataclass
class IngestionReport:
    """Report for a single entity ingestion."""

    entity_name: str
    source_file: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    rows_loaded: int = 0
    rows_valid: int = 0
    rows_invalid: int = 0
    rows_inserted: int = 0
    rows_skipped: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.rows_inserted > 0

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        duration = ""
        if self.completed_at:
            elapsed = (self.completed_at - self.started_at).total_seconds()
            duration = f" ({elapsed:.1f}s)"

        lines = [
            f"[{status}] {self.entity_name}{duration}",
            f"  Source: {self.source_file}",
            f"  Loaded: {self.rows_loaded} | Valid: {self.rows_valid} | "
            f"Invalid: {self.rows_invalid} | Inserted: {self.rows_inserted}",
        ]
        if self.errors:
            lines.append(f"  Errors: {'; '.join(self.errors[:3])}")
        return "\n".join(lines)


# =============================================================================
# SQL Insert Templates
# =============================================================================
# Each entity maps to its INSERT SQL and the columns to extract from validated data.
# Uses %s placeholders for psycopg2.

INSERT_SQL = {
    "regions": {
        "sql": """
            INSERT INTO sciro.regions (name, country)
            VALUES (%s, %s)
            ON CONFLICT (name) DO NOTHING
        """,
        "columns": ["name", "country"],
    },
    "suppliers": {
        "sql": """
            INSERT INTO sciro.suppliers (name, contact_email, contact_phone, region_id,
                                         lead_time_days, reliability_score, status)
            VALUES (%s, %s, %s,
                    (SELECT id FROM sciro.regions WHERE name = %s),
                    %s, %s, %s)
            ON CONFLICT (name) DO NOTHING
        """,
        "columns": [
            "name", "contact_email", "contact_phone", "region_name",
            "lead_time_days", "reliability_score", "status",
        ],
    },
    "products": {
        "sql": """
            INSERT INTO sciro.products (sku, name, category, unit_price, cost_price,
                                        weight_kg, reorder_point, safety_stock,
                                        primary_supplier_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    (SELECT id FROM sciro.suppliers WHERE name = %s LIMIT 1),
                    %s)
            ON CONFLICT (sku) DO NOTHING
        """,
        "columns": [
            "sku", "name", "category", "unit_price", "cost_price",
            "weight_kg", "reorder_point", "safety_stock", "supplier_name", "status",
        ],
    },
    "inventory": {
        "sql": """
            INSERT INTO sciro.inventory (product_id, region_id, quantity_on_hand, quantity_reserved)
            VALUES (
                (SELECT id FROM sciro.products WHERE sku = %s),
                (SELECT id FROM sciro.regions WHERE name = %s),
                %s, %s
            )
            ON CONFLICT (product_id, region_id) DO UPDATE SET
                quantity_on_hand = EXCLUDED.quantity_on_hand,
                quantity_reserved = EXCLUDED.quantity_reserved,
                updated_at = CURRENT_TIMESTAMP
        """,
        "columns": ["product_sku", "region_name", "quantity_on_hand", "quantity_reserved"],
    },
    "shipments": {
        "sql": """
            INSERT INTO sciro.shipments (
                tracking_number, supplier_id, product_id, destination_region_id,
                quantity, status, carrier, shipping_cost,
                shipped_date, expected_delivery_date, actual_delivery_date
            )
            VALUES (
                %s,
                (SELECT id FROM sciro.suppliers WHERE name = %s LIMIT 1),
                (SELECT id FROM sciro.products WHERE sku = %s LIMIT 1),
                (SELECT id FROM sciro.regions WHERE name = %s),
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (tracking_number) DO NOTHING
        """,
        "columns": [
            "tracking_number", "supplier_name", "product_sku", "destination_region",
            "quantity", "status", "carrier", "shipping_cost",
            "shipped_date", "expected_delivery_date", "actual_delivery_date",
        ],
    },
    "customers": {
        "sql": """
            INSERT INTO sciro.customers (name, email, company_name, region_id, tier)
            VALUES (%s, %s, %s,
                    (SELECT id FROM sciro.regions WHERE name = %s),
                    %s)
            ON CONFLICT (email) DO NOTHING
        """,
        "columns": ["name", "email", "company_name", "region_name", "tier"],
    },
    "orders": {
        "sql": """
            INSERT INTO sciro.orders (
                order_number, customer_id, region_id,
                order_date, status, payment_status, total_amount
            )
            VALUES (
                %s,
                (SELECT id FROM sciro.customers WHERE email = %s),
                (SELECT id FROM sciro.regions WHERE name = %s),
                %s, %s, %s, %s
            )
            ON CONFLICT (order_number) DO NOTHING
        """,
        "columns": [
            "order_number", "customer_email", "region_name",
            "order_date", "status", "payment_status", "total_amount",
        ],
    },
    "order_items": {
        "sql": """
            INSERT INTO sciro.order_items (
                order_id, product_id, quantity, unit_price, unit_cost, discount, line_total
            )
            VALUES (
                (SELECT id FROM sciro.orders WHERE order_number = %s),
                (SELECT id FROM sciro.products WHERE sku = %s),
                %s, %s,
                (SELECT cost_price FROM sciro.products WHERE sku = %s),
                %s,
                %s
            )
            ON CONFLICT (order_id, product_id) DO NOTHING
        """,
        "columns": [
            "order_number", "product_sku", "quantity", "unit_price",
            "product_sku", "discount", "line_total",
        ],
    },
}


def _compute_line_total(record: dict) -> dict:
    """Add computed line_total to order_items records."""
    if "line_total" not in record:
        qty = record.get("quantity", 0)
        price = record.get("unit_price", 0)
        discount = record.get("discount", 0)
        record["line_total"] = round(qty * price - discount, 2)
    return record


# =============================================================================
# Core Pipeline
# =============================================================================
def ingest_entity(
    entity_name: str,
    filepath: str,
    batch_size: int = 500,
    skip_validation: bool = False,
) -> IngestionReport:
    """Ingest a single entity from a file into the database.

    Pipeline: Load → Validate → Clean → Insert

    Args:
        entity_name: Entity type (e.g., 'suppliers', 'products').
        filepath: Path to the CSV/Excel file.
        batch_size: Number of records per INSERT batch.
        skip_validation: If True, skip Pydantic validation (for trusted data).

    Returns:
        IngestionReport with results.
    """
    from sciro.database.connection import get_connection

    report = IngestionReport(entity_name=entity_name, source_file=filepath)

    # Step 1: Load file
    logger.info("Loading %s from %s", entity_name, filepath)
    try:
        df = load_file(filepath)
        report.rows_loaded = len(df)
    except Exception as e:
        report.errors.append(f"Load failed: {e}")
        report.completed_at = datetime.now()
        return report

    if df.empty:
        report.errors.append("File is empty")
        report.completed_at = datetime.now()
        return report

    # Step 2: Validate
    if not skip_validation:
        logger.info("Validating %d rows for %s", len(df), entity_name)
        valid_records, error_records = validate_dataframe(df, entity_name)
        report.rows_valid = len(valid_records)
        report.rows_invalid = len(error_records)

        if not valid_records:
            report.errors.append("No valid records after validation")
            report.completed_at = datetime.now()
            return report
    else:
        valid_records = df.to_dict("records")
        report.rows_valid = len(valid_records)

    # Step 3: Compute derived fields
    if entity_name == "order_items":
        valid_records = [_compute_line_total(r) for r in valid_records]

    # Step 4: Insert into database
    if entity_name not in INSERT_SQL:
        report.errors.append(f"No INSERT SQL defined for '{entity_name}'")
        report.completed_at = datetime.now()
        return report

    insert_config = INSERT_SQL[entity_name]
    sql = insert_config["sql"]
    columns = insert_config["columns"]

    logger.info("Inserting %d records into %s", len(valid_records), entity_name)

    inserted = 0
    skipped = 0

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(valid_records), batch_size):
                    batch = valid_records[i : i + batch_size]
                    for record in batch:
                        try:
                            values = tuple(record.get(col) for col in columns)
                            cur.execute(sql, values)
                            if cur.rowcount > 0:
                                inserted += 1
                            else:
                                skipped += 1
                        except Exception as e:
                            skipped += 1
                            if skipped <= 5:
                                logger.warning(
                                    "Insert error for %s row: %s",
                                    entity_name, str(e)[:200],
                                )

                    logger.debug(
                        "Batch %d-%d: %d inserted",
                        i, min(i + batch_size, len(valid_records)), inserted,
                    )

    except Exception as e:
        report.errors.append(f"Database error: {e}")

    report.rows_inserted = inserted
    report.rows_skipped = skipped
    report.completed_at = datetime.now()

    logger.info(report.summary())
    return report


# =============================================================================
# Full Pipeline
# =============================================================================

# Load order matters: parents before children (FK dependencies)
LOAD_ORDER = [
    "regions",
    "suppliers",
    "products",
    "inventory",
    "shipments",
    "customers",
    "orders",
    "order_items",
]


def run_full_pipeline(
    data_dir: Optional[str] = None,
    entities: Optional[List[str]] = None,
) -> Dict[str, IngestionReport]:
    """Run the full ingestion pipeline for all entities.

    Loads files from data_dir in dependency order.

    Args:
        data_dir: Directory containing CSV files. Defaults to data/sample/.
        entities: Optional list of specific entities to load (must be in correct order).

    Returns:
        Dict mapping entity_name → IngestionReport.
    """
    from sciro.config.logging_config import setup_logging
    setup_logging()

    if data_dir is None:
        settings = get_settings()
        data_dir = str(settings.SAMPLE_DATA_DIR)

    entities_to_load = entities or LOAD_ORDER
    reports: Dict[str, IngestionReport] = {}

    logger.info("=" * 60)
    logger.info("SCIRO Data Ingestion Pipeline")
    logger.info("Data directory: %s", data_dir)
    logger.info("Entities to load: %s", entities_to_load)
    logger.info("=" * 60)

    for entity in entities_to_load:
        filepath = os.path.join(data_dir, f"{entity}.csv")

        if not os.path.exists(filepath):
            logger.warning("File not found for %s: %s — skipping", entity, filepath)
            reports[entity] = IngestionReport(
                entity_name=entity,
                source_file=filepath,
                errors=[f"File not found: {filepath}"],
                completed_at=datetime.now(),
            )
            continue

        report = ingest_entity(entity, filepath)
        reports[entity] = report

        if not report.success:
            logger.error(
                "Failed to ingest %s. Downstream entities may be affected.",
                entity,
            )

    # Print final summary
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    for entity, report in reports.items():
        logger.info(report.summary())
    logger.info("=" * 60)

    return reports


# =============================================================================
# CLI Entry Point
# =============================================================================
def main() -> None:
    """Entry point: python -m sciro.ingestion.pipeline"""
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    reports = run_full_pipeline(data_dir=data_dir)

    # Exit with error if any entity failed
    failed = [name for name, r in reports.items() if not r.success]
    if failed:
        print(f"\nFailed entities: {failed}")
        sys.exit(1)
    else:
        print("\nAll entities ingested successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
