"""
Tests for SCIRO Customer Analytics.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.analytics.customer import (
    get_customer_metrics,
    calculate_customer_ltv,
    segment_customers,
    calculate_churn_probability,
    get_customer_summary,
)


@pytest.fixture
def sample_customer_data():
    customers = pd.DataFrame([
        {
            "name": "Alice Smith",
            "email": "alice@example.com",
            "company_name": "Alice Corp",
            "region_name": "North",
            "tier": "enterprise",
        },
        {
            "name": "Bob Jones",
            "email": "bob@example.com",
            "company_name": "Bob LLC",
            "region_name": "South",
            "tier": "standard",
        },
    ])

    orders = pd.DataFrame([
        {
            "order_number": "ORD-001",
            "customer_email": "alice@example.com",
            "region_name": "North",
            "order_date": "2026-01-01",
            "status": "delivered",
            "total_amount": 1000.0,
        },
        {
            "order_number": "ORD-002",
            "customer_email": "alice@example.com",
            "region_name": "North",
            "order_date": "2026-02-01",
            "status": "delivered",
            "total_amount": 1500.0,
        },
        {
            "order_number": "ORD-003",
            "customer_email": "bob@example.com",
            "region_name": "South",
            "order_date": "2025-06-01",  # Old order
            "status": "delivered",
            "total_amount": 500.0,
        },
    ])

    order_items = pd.DataFrame([
        {"order_number": "ORD-001", "product_sku": "P1", "quantity": 1, "unit_price": 1000.0, "discount": 0.0},
        {"order_number": "ORD-002", "product_sku": "P1", "quantity": 1, "unit_price": 1500.0, "discount": 0.0},
        {"order_number": "ORD-003", "product_sku": "P2", "quantity": 1, "unit_price": 500.0, "discount": 0.0},
    ])

    return customers, orders, order_items


class TestCustomerAnalytics:
    def test_get_customer_metrics(self, sample_customer_data):
        customers, orders, order_items = sample_customer_data
        metrics = get_customer_metrics(customers, orders, order_items)

        assert len(metrics) == 2
        alice = metrics[metrics["email"] == "alice@example.com"].iloc[0]
        assert alice["total_orders"] == 2
        assert alice["total_revenue"] == 2500.0
        assert alice["avg_order_value"] == 1250.0

    def test_calculate_customer_ltv(self, sample_customer_data):
        customers, orders, order_items = sample_customer_data
        ltv = calculate_customer_ltv(customers, orders, order_items)

        assert len(ltv) == 2
        assert "estimated_clv" in ltv.columns
        assert "clv_tier" in ltv.columns
        alice = ltv[ltv["email"] == "alice@example.com"].iloc[0]
        assert alice["estimated_clv"] > 0

    def test_segment_customers(self, sample_customer_data):
        customers, orders, order_items = sample_customer_data
        segmented = segment_customers(customers, orders, order_items)

        assert len(segmented) == 2
        assert "segment" in segmented.columns
        assert "rfm_score" in segmented.columns

    def test_calculate_churn_probability(self, sample_customer_data):
        customers, orders, _ = sample_customer_data
        churn = calculate_churn_probability(customers, orders)

        assert len(churn) == 2
        assert "churn_score" in churn.columns
        assert "churn_risk" in churn.columns

        alice = churn[churn["email"] == "alice@example.com"].iloc[0]
        bob = churn[churn["email"] == "bob@example.com"].iloc[0]

        # Bob's order was much older than Alice's
        assert bob["churn_score"] >= alice["churn_score"]

    def test_get_customer_summary(self, sample_customer_data):
        customers, orders, order_items = sample_customer_data
        summary = get_customer_summary(customers, orders, order_items)

        assert summary["total_customers"] == 2
        assert "customers_by_tier" in summary
        assert "customers_by_segment" in summary
