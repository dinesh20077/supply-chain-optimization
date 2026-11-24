"""
Tests for SCIRO Supply Chain Analytics.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.analytics.supply_chain import (
    get_supplier_performance,
    calculate_supplier_risk_scores,
    get_shipment_anomalies,
    get_supplier_dependency,
    get_supply_chain_summary,
)


@pytest.fixture
def sample_sc_data():
    suppliers = pd.DataFrame([
        {
            "name": "Supplier A",
            "region_name": "North",
            "status": "active",
            "lead_time_days": 5,
            "reliability_score": 0.95,
        },
        {
            "name": "Supplier B",
            "region_name": "South",
            "status": "active",
            "lead_time_days": 10,
            "reliability_score": 0.60,
        },
    ])

    products = pd.DataFrame([
        {"sku": "P1", "supplier_name": "Supplier A"},
        {"sku": "P2", "supplier_name": "Supplier A"},
        {"sku": "P3", "supplier_name": "Supplier B"},
    ])

    shipments = pd.DataFrame([
        # Supplier A: on-time delivered
        {
            "tracking_number": "TRK-001",
            "supplier_name": "Supplier A",
            "product_sku": "P1",
            "destination_region": "North",
            "status": "delivered",
            "shipping_cost": 500.0,
            "shipped_date": "2026-01-01",
            "expected_delivery_date": "2026-01-06",
            "actual_delivery_date": "2026-01-05",
        },
        # Supplier B: delayed delivered
        {
            "tracking_number": "TRK-002",
            "supplier_name": "Supplier B",
            "product_sku": "P3",
            "destination_region": "South",
            "status": "delivered",
            "shipping_cost": 800.0,
            "shipped_date": "2026-01-01",
            "expected_delivery_date": "2026-01-11",
            "actual_delivery_date": "2026-01-20",
        },
        # Supplier B: currently delayed
        {
            "tracking_number": "TRK-003",
            "supplier_name": "Supplier B",
            "product_sku": "P3",
            "destination_region": "South",
            "status": "delayed",
            "shipping_cost": 750.0,
            "shipped_date": "2026-01-10",
            "expected_delivery_date": "2026-01-20",
            "actual_delivery_date": None,
        },
    ])

    return suppliers, shipments, products


class TestSupplyChainAnalytics:
    def test_supplier_performance_calculation(self, sample_sc_data):
        suppliers, shipments, products = sample_sc_data
        perf = get_supplier_performance(suppliers, shipments, products)

        assert len(perf) == 2
        sup_a = perf[perf["supplier_name"] == "Supplier A"].iloc[0]
        sup_b = perf[perf["supplier_name"] == "Supplier B"].iloc[0]

        assert sup_a["on_time_delivery_pct"] == 100.0
        assert sup_a["late_shipments"] == 0
        assert sup_a["products_supplied"] == 2

        assert sup_b["on_time_delivery_pct"] == 0.0
        assert sup_b["late_shipments"] == 1
        assert sup_b["currently_delayed"] == 1
        assert sup_b["products_supplied"] == 1

    def test_supplier_risk_scoring(self, sample_sc_data):
        suppliers, shipments, products = sample_sc_data
        perf = get_supplier_performance(suppliers, shipments, products)
        scored = calculate_supplier_risk_scores(perf)

        assert "risk_score" in scored.columns
        assert "risk_level" in scored.columns

        sup_a = scored[scored["supplier_name"] == "Supplier A"].iloc[0]
        sup_b = scored[scored["supplier_name"] == "Supplier B"].iloc[0]

        # Supplier B should have higher risk than Supplier A
        assert sup_b["risk_score"] > sup_a["risk_score"]

    def test_supplier_dependency(self, sample_sc_data):
        suppliers, shipments, products = sample_sc_data
        dep = get_supplier_dependency(products)

        assert len(dep) == 2
        assert "product_pct" in dep.columns
        # Supplier A has 2/3 products (66.67%)
        sup_a = dep[dep["supplier_name"] == "Supplier A"].iloc[0]
        assert sup_a["products_count"] == 2
        assert sup_a["product_pct"] == pytest.approx(66.67, rel=1e-2)

    def test_supply_chain_summary(self, sample_sc_data):
        suppliers, shipments, products = sample_sc_data
        summary = get_supply_chain_summary(suppliers, shipments, products)

        assert summary["total_suppliers"] == 2
        assert summary["total_shipments"] == 3
        assert summary["delivered_shipments"] == 2
        assert summary["delayed_shipments"] == 1
        assert "top_risk_suppliers" in summary
