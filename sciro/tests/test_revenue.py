"""
Tests for SCIRO Revenue Analytics.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.analytics.revenue import (
    get_revenue_summary,
    get_revenue_growth,
    get_revenue_by_product,
    get_revenue_by_region,
    get_revenue_by_customer,
    get_product_performance,
    get_monthly_revenue_trend,
)


@pytest.fixture
def sample_revenue_data():
    orders = pd.DataFrame([
        {
            "order_number": "ORD-001",
            "customer_email": "alice@example.com",
            "region_name": "North",
            "order_date": "2026-02-01",
            "status": "delivered",
            "payment_status": "paid",
            "total_amount": 2000.0,
        },
        {
            "order_number": "ORD-002",
            "customer_email": "bob@example.com",
            "region_name": "South",
            "order_date": "2026-02-20",
            "status": "delivered",
            "payment_status": "paid",
            "total_amount": 1000.0,
        },
        {
            "order_number": "ORD-003",
            "customer_email": "charlie@example.com",
            "region_name": "North",
            "order_date": "2026-01-05",
            "status": "cancelled",  # Cancelled order should be excluded from revenue
            "payment_status": "unpaid",
            "total_amount": 5000.0,
        },
    ])

    order_items = pd.DataFrame([
        {"order_number": "ORD-001", "product_sku": "P1", "quantity": 2, "unit_price": 1000.0, "discount": 0.0},
        {"order_number": "ORD-002", "product_sku": "P2", "quantity": 2, "unit_price": 500.0, "discount": 0.0},
        {"order_number": "ORD-003", "product_sku": "P1", "quantity": 5, "unit_price": 1000.0, "discount": 0.0},
    ])

    products = pd.DataFrame([
        {
            "sku": "P1",
            "name": "Laptop",
            "category": "Electronics",
            "unit_price": 1000.0,
            "cost_price": 700.0,
            "weight_kg": 2.0,
            "reorder_point": 10,
            "safety_stock": 5,
            "supplier_name": "Supplier A",
            "status": "active",
        },
        {
            "sku": "P2",
            "name": "Mouse",
            "category": "Electronics",
            "unit_price": 500.0,
            "cost_price": 300.0,
            "weight_kg": 0.2,
            "reorder_point": 20,
            "safety_stock": 10,
            "supplier_name": "Supplier B",
            "status": "active",
        },
    ])

    return orders, order_items, products


class TestRevenueAnalytics:
    def test_revenue_summary(self, sample_revenue_data):
        orders, order_items, products = sample_revenue_data
        summary = get_revenue_summary(orders, order_items, products, days=60)

        # Non-cancelled orders: ORD-001 (2000) + ORD-002 (1000) = 3000
        assert summary["total_orders"] == 2
        assert summary["total_revenue"] == 3000.0
        assert summary["avg_order_value"] == 1500.0
        assert summary["total_units_sold"] == 4
        # COGS: P1 (2 * 700 = 1400) + P2 (2 * 300 = 600) = 2000
        assert summary["total_cogs"] == 2000.0
        assert summary["gross_profit"] == 1000.0
        assert summary["gross_margin_pct"] == pytest.approx(33.33, rel=1e-2)

    def test_revenue_by_product(self, sample_revenue_data):
        orders, order_items, products = sample_revenue_data
        by_prod = get_revenue_by_product(orders, order_items, products, days=60)

        assert len(by_prod) == 2
        assert by_prod.iloc[0]["product_sku"] == "P1"
        assert by_prod.iloc[0]["revenue"] == 2000.0

    def test_revenue_by_region(self, sample_revenue_data):
        orders, order_items, products = sample_revenue_data
        by_region = get_revenue_by_region(orders, order_items, days=60)

        assert len(by_region) == 2
        north = by_region[by_region["region_name"] == "North"].iloc[0]
        assert north["revenue"] == 2000.0

    def test_revenue_by_customer(self, sample_revenue_data):
        orders, order_items, products = sample_revenue_data
        by_cust = get_revenue_by_customer(orders, order_items)

        assert len(by_cust) == 2
        assert "alice@example.com" in by_cust["customer_email"].values
        assert "charlie@example.com" not in by_cust["customer_email"].values

    def test_product_performance(self, sample_revenue_data):
        orders, order_items, products = sample_revenue_data
        perf = get_product_performance(orders, order_items, products)

        assert len(perf) == 2
        assert "revenue" in perf.columns
        assert "profit" in perf.columns
        assert "margin_pct" in perf.columns
