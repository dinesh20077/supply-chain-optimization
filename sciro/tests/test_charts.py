"""
Unit tests for SCIRO Dashboard Charting Library.
"""

import pytest
import pandas as pd
import numpy as np

from sciro.dashboard.charts import (
    create_revenue_trend_chart,
    create_revenue_at_risk_donut,
    create_product_revenue_bar,
    create_regional_revenue_chart,
    create_supplier_risk_distribution,
    create_supplier_scatter,
    create_inventory_health_donut,
    create_stockout_risk_scatter,
    create_customer_segment_treemap,
    create_customer_ltv_bar,
    create_feature_importance_chart,
    create_revenue_forecast_chart,
    create_monte_carlo_histogram,
)


class TestDashboardCharts:
    def test_revenue_trend_chart(self):
        df = pd.DataFrame({
            "month": ["2026-01", "2026-02", "2026-03"],
            "revenue": [100000.0, 150000.0, 180000.0],
            "orders": [120, 140, 175],
        })
        fig = create_revenue_trend_chart(df)
        assert fig is not None
        assert len(fig.data) == 2

    def test_revenue_trend_empty(self):
        fig = create_revenue_trend_chart(pd.DataFrame())
        assert fig is not None

    def test_revenue_at_risk_donut(self):
        risk_data = {
            "total_revenue_at_risk": 500000.0,
            "supplier_risks": [
                {"supplier_name": "Sup1", "risk_level": "HIGH", "revenue_at_risk": 300000.0},
                {"supplier_name": "Sup2", "risk_level": "MEDIUM", "revenue_at_risk": 200000.0},
            ],
        }
        fig = create_revenue_at_risk_donut(risk_data)
        assert fig is not None
        assert len(fig.data) == 1

    def test_product_revenue_bar(self):
        df = pd.DataFrame({
            "name": ["Prod A", "Prod B", "Prod C"],
            "revenue": [50000.0, 75000.0, 25000.0],
        })
        fig = create_product_revenue_bar(df, top_n=2)
        assert fig is not None

    def test_regional_revenue_chart(self):
        df = pd.DataFrame({
            "region_name": ["North", "South", "West"],
            "revenue": [100000.0, 200000.0, 150000.0],
        })
        fig = create_regional_revenue_chart(df)
        assert fig is not None

    def test_supplier_risk_distribution(self):
        df = pd.DataFrame({"risk_level": ["LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"]})
        fig = create_supplier_risk_distribution(df)
        assert fig is not None

    def test_supplier_scatter(self):
        df = pd.DataFrame({
            "supplier_name": ["S1", "S2"],
            "on_time_delivery_pct": [95.0, 78.0],
            "avg_delay_days": [1.2, 5.4],
            "risk_level": ["LOW", "HIGH"],
            "risk_score": [20.0, 80.0],
        })
        fig = create_supplier_scatter(df)
        assert fig is not None

    def test_inventory_health_donut(self):
        df = pd.DataFrame({"stock_status": ["Healthy", "Healthy", "Critical Shortage", "Out of Stock"]})
        fig = create_inventory_health_donut(df)
        assert fig is not None

    def test_stockout_risk_scatter(self):
        df = pd.DataFrame({
            "product_sku": ["P1", "P2"],
            "name": ["Prod 1", "Prod 2"],
            "days_of_supply": [12.0, 3.5],
            "stockout_probability": [0.2, 0.85],
            "stockout_risk": ["LOW", "HIGH"],
        })
        fig = create_stockout_risk_scatter(df)
        assert fig is not None

    def test_customer_segment_treemap(self):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "segment": ["Champions", "Champions", "At Risk", "Loyal Customers"],
            "monetary": [1000.0, 2000.0, 500.0, 1500.0],
        })
        fig = create_customer_segment_treemap(df)
        assert fig is not None

    def test_customer_ltv_bar(self):
        df = pd.DataFrame({"clv_tier": ["Platinum", "Gold", "Silver", "Bronze", "Gold"]})
        fig = create_customer_ltv_bar(df)
        assert fig is not None

    def test_feature_importance_chart(self):
        df = pd.DataFrame({
            "feature": ["lag_1", "rolling_mean", "price"],
            "importance": [0.45, 0.35, 0.20],
        })
        fig = create_feature_importance_chart(df)
        assert fig is not None

    def test_revenue_forecast_chart(self):
        df = pd.DataFrame({
            "month": ["2026-04", "2026-05"],
            "predicted_revenue": [210000.0, 230000.0],
        })
        fig = create_revenue_forecast_chart(df)
        assert fig is not None

    def test_monte_carlo_histogram(self):
        losses = [10000.0, 15000.0, 22000.0, 18000.0, 35000.0]
        fig = create_monte_carlo_histogram(losses, p95_loss=32000.0, mean_loss=20000.0)
        assert fig is not None
