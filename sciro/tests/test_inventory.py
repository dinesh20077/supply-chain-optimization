"""
Tests for SCIRO Inventory Analytics.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.analytics.inventory import (
    calculate_avg_daily_demand,
    get_inventory_status,
    calculate_stockout_probability,
    calculate_inventory_turnover,
    get_inventory_summary,
)


@pytest.fixture
def sample_inventory_data():
    products = pd.DataFrame([
        {
            "sku": "P1",
            "name": "Widget A",
            "category": "Electronics",
            "unit_price": 1000.0,
            "cost_price": 600.0,
            "reorder_point": 20,
            "safety_stock": 10,
            "supplier_name": "Supplier A",
        },
        {
            "sku": "P2",
            "name": "Widget B",
            "category": "Electronics",
            "unit_price": 500.0,
            "cost_price": 300.0,
            "reorder_point": 50,
            "safety_stock": 25,
            "supplier_name": "Supplier B",
        },
    ])

    inventory = pd.DataFrame([
        {
            "product_sku": "P1",
            "region_name": "North",
            "quantity_on_hand": 5,
            "quantity_reserved": 0,
        },
        {
            "product_sku": "P2",
            "region_name": "North",
            "quantity_on_hand": 200,
            "quantity_reserved": 10,
        },
    ])

    orders = pd.DataFrame([
        {
            "order_number": "ORD-001",
            "order_date": "2026-02-01",
            "status": "delivered",
        },
        {
            "order_number": "ORD-002",
            "order_date": "2026-02-15",
            "status": "delivered",
        },
    ])

    order_items = pd.DataFrame([
        {"order_number": "ORD-001", "product_sku": "P1", "quantity": 10, "unit_price": 1000.0, "discount": 0},
        {"order_number": "ORD-002", "product_sku": "P1", "quantity": 20, "unit_price": 1000.0, "discount": 0},
        {"order_number": "ORD-001", "product_sku": "P2", "quantity": 5, "unit_price": 500.0, "discount": 0},
    ])

    suppliers = pd.DataFrame([
        {"name": "Supplier A", "reliability_score": 0.8, "lead_time_days": 7},
        {"name": "Supplier B", "reliability_score": 0.95, "lead_time_days": 3},
    ])

    return products, inventory, orders, order_items, suppliers


class TestInventoryAnalytics:
    def test_calculate_avg_daily_demand(self, sample_inventory_data):
        products, inventory, orders, order_items, _ = sample_inventory_data
        demand = calculate_avg_daily_demand(orders, order_items, days=30)

        assert len(demand) == 2
        p1_demand = demand[demand["product_sku"] == "P1"].iloc[0]
        # P1 sold 30 units across 30 days => 1.0 unit/day
        assert p1_demand["total_quantity_sold"] == 30
        assert p1_demand["avg_daily_demand"] == 1.0

    def test_inventory_status(self, sample_inventory_data):
        products, inventory, orders, order_items, suppliers = sample_inventory_data
        status = get_inventory_status(inventory, products, orders, order_items, suppliers)

        assert len(status) == 2
        p1 = status[status["product_sku"] == "P1"].iloc[0]
        p2 = status[status["product_sku"] == "P2"].iloc[0]

        # P1 has 5 units available, safety stock is 10 => CRITICAL
        assert p1["stock_status"] == "CRITICAL"
        assert p1["quantity_available"] == 5

        # P2 has 190 units available, reorder point 50 => HEALTHY
        assert p2["stock_status"] == "HEALTHY"
        assert p2["quantity_available"] == 190

    def test_stockout_probability(self, sample_inventory_data):
        products, inventory, orders, order_items, suppliers = sample_inventory_data
        status = get_inventory_status(inventory, products, orders, order_items, suppliers)
        stockout = calculate_stockout_probability(status)

        assert "stockout_probability" in stockout.columns
        assert "stockout_risk" in stockout.columns

        p1 = stockout[stockout["product_sku"] == "P1"].iloc[0]
        p2 = stockout[stockout["product_sku"] == "P2"].iloc[0]

        # P1 low stock should have higher stockout probability than P2 ample stock
        assert p1["stockout_probability"] > p2["stockout_probability"]

    def test_inventory_turnover(self, sample_inventory_data):
        products, inventory, orders, order_items, _ = sample_inventory_data
        turnover = calculate_inventory_turnover(inventory, products, orders, order_items, days=30)

        assert len(turnover) == 2
        assert "turnover_ratio" in turnover.columns
        assert "annualized_turnover" in turnover.columns

    def test_inventory_summary(self, sample_inventory_data):
        products, inventory, orders, order_items, suppliers = sample_inventory_data
        summary = get_inventory_summary(inventory, products, orders, order_items, suppliers)

        assert summary["total_inventory_records"] == 2
        assert summary["critical_stock_count"] >= 1
        assert "total_potential_lost_revenue" in summary
