"""
Tests for SCIRO Revenue-at-Risk Engine.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.analytics.risk_engine import (
    calculate_revenue_at_risk_by_supplier,
    get_risk_dashboard_data,
    get_supplier_impact_report,
    explain_risk_chain,
)


@pytest.fixture
def sample_risk_data():
    suppliers = pd.DataFrame([
        {
            "name": "Supplier A",
            "region_name": "North",
            "status": "active",
            "lead_time_days": 15,
            "reliability_score": 0.40,  # Unreliable supplier
        },
        {
            "name": "Supplier B",
            "region_name": "South",
            "status": "active",
            "lead_time_days": 3,
            "reliability_score": 0.98,  # Highly reliable supplier
        },
    ])

    products = pd.DataFrame([
        {
            "sku": "P1",
            "name": "Critical Widget",
            "category": "Electronics",
            "unit_price": 2000.0,
            "cost_price": 1200.0,
            "reorder_point": 50,
            "safety_stock": 20,
            "supplier_name": "Supplier A",
            "status": "active",
        },
        {
            "sku": "P2",
            "name": "Safe Widget",
            "category": "Raw Materials",
            "unit_price": 500.0,
            "cost_price": 300.0,
            "reorder_point": 20,
            "safety_stock": 10,
            "supplier_name": "Supplier B",
            "status": "active",
        },
    ])

    inventory = pd.DataFrame([
        {
            "product_sku": "P1",
            "region_name": "North",
            "quantity_on_hand": 0,  # Out of stock
            "quantity_reserved": 0,
        },
        {
            "product_sku": "P2",
            "region_name": "South",
            "quantity_on_hand": 500,  # Plentiful stock
            "quantity_reserved": 0,
        },
    ])

    orders = pd.DataFrame([
        {
            "order_number": "ORD-001",
            "customer_email": "vip@enterprise.com",
            "region_name": "North",
            "order_date": "2026-02-01",
            "status": "delivered",
            "total_amount": 100000.0,
        },
        {
            "order_number": "ORD-002",
            "customer_email": "regular@company.com",
            "region_name": "South",
            "order_date": "2026-02-15",
            "status": "delivered",
            "total_amount": 5000.0,
        },
    ])

    order_items = pd.DataFrame([
        {"order_number": "ORD-001", "product_sku": "P1", "quantity": 50, "unit_price": 2000.0, "discount": 0.0},
        {"order_number": "ORD-002", "product_sku": "P2", "quantity": 10, "unit_price": 500.0, "discount": 0.0},
    ])

    shipments = pd.DataFrame([
        # Supplier A: late shipment
        {
            "tracking_number": "TRK-001",
            "supplier_name": "Supplier A",
            "product_sku": "P1",
            "destination_region": "North",
            "quantity": 100,
            "status": "delivered",
            "shipping_cost": 1000.0,
            "shipped_date": "2026-01-01",
            "expected_delivery_date": "2026-01-16",
            "actual_delivery_date": "2026-01-28",  # 12 days late!
        },
        # Supplier B: on-time shipment
        {
            "tracking_number": "TRK-002",
            "supplier_name": "Supplier B",
            "product_sku": "P2",
            "destination_region": "South",
            "quantity": 200,
            "status": "delivered",
            "shipping_cost": 500.0,
            "shipped_date": "2026-01-01",
            "expected_delivery_date": "2026-01-04",
            "actual_delivery_date": "2026-01-04",  # On time
        },
    ])

    return suppliers, products, inventory, orders, order_items, shipments


class TestRevenueAtRiskEngine:
    def test_calculate_revenue_at_risk_by_supplier(self, sample_risk_data):
        suppliers, products, inventory, orders, order_items, shipments = sample_risk_data
        rar_df = calculate_revenue_at_risk_by_supplier(
            suppliers, products, inventory, orders, order_items, shipments, horizon_days=30
        )

        assert len(rar_df) == 2
        sup_a = rar_df[rar_df["supplier_name"] == "Supplier A"].iloc[0]
        sup_b = rar_df[rar_df["supplier_name"] == "Supplier B"].iloc[0]

        # Supplier A should carry significantly higher revenue at risk
        assert sup_a["total_revenue_at_risk"] > sup_b["total_revenue_at_risk"]
        assert sup_a["products_at_risk"] >= 1
        assert len(sup_a["reasoning"]) > 0
        assert len(sup_a["recommendation"]) > 0

    def test_get_risk_dashboard_data(self, sample_risk_data):
        suppliers, products, inventory, orders, order_items, shipments = sample_risk_data
        dash = get_risk_dashboard_data(
            suppliers, products, inventory, orders, order_items, shipments
        )

        assert "total_revenue_at_risk" in dash
        assert dash["total_revenue_at_risk"] > 0
        assert "risk_distribution" in dash
        assert "top_risk_suppliers" in dash
        assert len(dash["top_risk_suppliers"]) > 0

    def test_get_supplier_impact_report(self, sample_risk_data):
        suppliers, products, inventory, orders, order_items, shipments = sample_risk_data
        report = get_supplier_impact_report(
            "Supplier A", suppliers, products, inventory, orders, order_items, shipments
        )

        assert report["supplier_name"] == "Supplier A"
        assert "risk_chain" in report
        assert "step_1_supplier_risk" in report["risk_chain"]
        assert "step_4_revenue_impact" in report["risk_chain"]
        assert report["revenue_at_risk"] > 0

    def test_explain_risk_chain(self, sample_risk_data):
        suppliers, products, inventory, orders, order_items, shipments = sample_risk_data
        rar_df = calculate_revenue_at_risk_by_supplier(
            suppliers, products, inventory, orders, order_items, shipments
        )

        explanation = explain_risk_chain("Supplier A", rar_df)
        assert "Risk Analysis: Supplier A" in explanation
        assert "Revenue at Risk: Rs" in explanation
        assert "Reasoning" in explanation
        assert "Recommendation" in explanation
