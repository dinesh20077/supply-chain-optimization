"""
Integration tests for SCIRO Ingestion & Analytics Pipeline.
"""

import os
import pytest
import pandas as pd

from sciro.ingestion.loader import load_file
from sciro.ingestion.validators import validate_dataframe
from sciro.etl.cleaning import clean_dataframe
from sciro.etl.quality import run_quality_checks
from sciro.analytics import (
    load_supply_chain_data,
    get_supplier_performance,
    calculate_supplier_risk_scores,
    get_inventory_status,
    calculate_stockout_probability,
    get_revenue_summary,
    segment_customers,
    calculate_revenue_at_risk_by_supplier,
    get_risk_dashboard_data,
)


class TestEndToEndPipeline:
    def test_load_and_validate_all_sample_files(self):
        sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
        entities = [
            "regions", "suppliers", "products", "inventory",
            "shipments", "customers", "orders", "order_items"
        ]

        for entity in entities:
            filepath = os.path.join(sample_dir, f"{entity}.csv")
            assert os.path.exists(filepath), f"File {filepath} does not exist"

            df = load_file(filepath)
            assert len(df) > 0, f"DataFrame for {entity} is empty"

            valid_records, errors = validate_dataframe(df, entity)
            assert len(errors) == 0, f"Found {len(errors)} validation errors in {entity}"
            assert len(valid_records) == len(df)

    def test_end_to_end_analytics_execution(self):
        sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")

        suppliers = load_file(os.path.join(sample_dir, "suppliers.csv"))
        products = load_file(os.path.join(sample_dir, "products.csv"))
        inventory = load_file(os.path.join(sample_dir, "inventory.csv"))
        shipments = load_file(os.path.join(sample_dir, "shipments.csv"))
        customers = load_file(os.path.join(sample_dir, "customers.csv"))
        orders = load_file(os.path.join(sample_dir, "orders.csv"))
        order_items = load_file(os.path.join(sample_dir, "order_items.csv"))

        # 1. Supply Chain Analytics
        perf = get_supplier_performance(suppliers, shipments, products)
        scored = calculate_supplier_risk_scores(perf)
        assert len(scored) == len(suppliers)
        assert "risk_score" in scored.columns

        # 2. Inventory Analytics
        inv_status = get_inventory_status(inventory, products, orders, order_items, suppliers)
        assert len(inv_status) == len(inventory)
        stockout = calculate_stockout_probability(inv_status)
        assert len(stockout) == len(inventory)

        # 3. Revenue Analytics
        rev_summary = get_revenue_summary(orders, order_items, products, days=365)
        assert rev_summary["total_revenue"] > 0
        assert rev_summary["total_orders"] > 0

        # 4. Customer Analytics
        segmented = segment_customers(customers, orders, order_items)
        assert len(segmented) == len(customers)

        # 5. Revenue-at-Risk Engine
        risk_df = calculate_revenue_at_risk_by_supplier(
            suppliers, products, inventory, orders, order_items, shipments
        )
        assert len(risk_df) > 0
        assert "total_revenue_at_risk" in risk_df.columns
        assert risk_df["total_revenue_at_risk"].sum() > 0

        dash = get_risk_dashboard_data(
            suppliers, products, inventory, orders, order_items, shipments
        )
        assert dash["total_revenue_at_risk"] > 0
        assert dash["high_risk_supplier_count"] >= 0
