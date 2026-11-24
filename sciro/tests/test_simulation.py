"""
Tests for SCIRO Simulation & Optimization (Module 4).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sim_data():
    """Generate test data for simulation module."""
    np.random.seed(42)
    base_date = datetime(2026, 1, 1)

    suppliers = pd.DataFrame([
        {"name": "Reliable Co", "region_name": "South", "status": "active",
         "lead_time_days": 5, "reliability_score": 0.95},
        {"name": "Risky Ltd", "region_name": "North", "status": "active",
         "lead_time_days": 15, "reliability_score": 0.40},
    ])

    products = pd.DataFrame([
        {"sku": "P1", "name": "Product Alpha", "category": "Electronics",
         "unit_price": 2000.0, "cost_price": 1200.0, "reorder_point": 50,
         "safety_stock": 20, "supplier_name": "Risky Ltd"},
        {"sku": "P2", "name": "Product Beta", "category": "Raw Materials",
         "unit_price": 500.0, "cost_price": 300.0, "reorder_point": 30,
         "safety_stock": 10, "supplier_name": "Reliable Co"},
    ])

    inventory = pd.DataFrame([
        {"product_sku": "P1", "region_name": "North",
         "quantity_on_hand": 25, "quantity_reserved": 5},
        {"product_sku": "P2", "region_name": "South",
         "quantity_on_hand": 300, "quantity_reserved": 20},
        {"product_sku": "P1", "region_name": "South",
         "quantity_on_hand": 100, "quantity_reserved": 0},
    ])

    # 60 days of orders
    orders_list = []
    items_list = []
    for i in range(60):
        date = base_date + timedelta(days=i)
        for j in range(np.random.randint(2, 5)):
            order_num = f"ORD-{i:03d}-{j:02d}"
            orders_list.append({
                "order_number": order_num,
                "customer_email": f"cust{np.random.randint(1,15)}@test.com",
                "region_name": np.random.choice(["North", "South"]),
                "order_date": date.strftime("%Y-%m-%d"),
                "status": np.random.choice(["delivered", "confirmed"], p=[0.8, 0.2]),
                "total_amount": np.random.uniform(500, 5000),
            })
            for sku in np.random.choice(["P1", "P2"], size=np.random.randint(1, 3), replace=False):
                prod = products[products["sku"] == sku].iloc[0]
                items_list.append({
                    "order_number": order_num,
                    "product_sku": sku,
                    "quantity": np.random.randint(1, 10),
                    "unit_price": prod["unit_price"],
                    "discount": 0,
                })

    orders = pd.DataFrame(orders_list)
    order_items = pd.DataFrame(items_list)

    return {
        "suppliers": suppliers, "products": products, "inventory": inventory,
        "orders": orders, "order_items": order_items,
    }


# =============================================================================
# Disruption Simulator Tests
# =============================================================================
class TestDisruptionSimulator:
    def test_basic_simulation(self, sim_data):
        from sciro.simulation.disruption_simulator import DisruptionSimulator, ScenarioConfig
        sim = DisruptionSimulator()
        config = ScenarioConfig(
            name="test_basic",
            num_simulations=100,
            horizon_days=14,
        )
        result = sim.run_simulation(
            config, sim_data["products"], sim_data["inventory"],
            sim_data["orders"], sim_data["order_items"], sim_data["suppliers"],
        )
        assert result.scenario == "test_basic"
        assert result.num_simulations == 100
        assert result.revenue_at_risk_mean >= 0
        assert result.revenue_at_risk_p95 >= result.revenue_at_risk_mean

    def test_supplier_failure(self, sim_data):
        from sciro.simulation.disruption_simulator import DisruptionSimulator, ScenarioConfig
        sim = DisruptionSimulator()
        config = ScenarioConfig(
            name="supplier_failure",
            failed_suppliers=["Risky Ltd"],
            num_simulations=100,
            horizon_days=30,
        )
        result = sim.run_simulation(
            config, sim_data["products"], sim_data["inventory"],
            sim_data["orders"], sim_data["order_items"], sim_data["suppliers"],
        )
        assert result.revenue_at_risk_mean > 0
        assert len(result.summary) > 0

    def test_demand_surge(self, sim_data):
        from sciro.simulation.disruption_simulator import DisruptionSimulator, ScenarioConfig
        sim = DisruptionSimulator()
        config = ScenarioConfig(
            name="demand_surge",
            demand_surge_pct=0.50,
            num_simulations=100,
            horizon_days=14,
        )
        result = sim.run_simulation(
            config, sim_data["products"], sim_data["inventory"],
            sim_data["orders"], sim_data["order_items"], sim_data["suppliers"],
        )
        assert result.revenue_at_risk_mean >= 0

    def test_scenario_comparison(self, sim_data):
        from sciro.simulation.disruption_simulator import DisruptionSimulator, ScenarioConfig
        sim = DisruptionSimulator()
        scenarios = [
            ScenarioConfig(name="baseline", num_simulations=50, horizon_days=14),
            ScenarioConfig(name="surge", demand_surge_pct=0.5, num_simulations=50, horizon_days=14),
        ]
        comparison = sim.run_scenario_comparison(
            scenarios, sim_data["products"], sim_data["inventory"],
            sim_data["orders"], sim_data["order_items"], sim_data["suppliers"],
        )
        assert len(comparison) == 2
        assert "scenario" in comparison.columns


# =============================================================================
# Pricing Optimizer Tests
# =============================================================================
class TestPricingOptimizer:
    def test_estimate_elasticity(self, sim_data):
        from sciro.simulation.pricing_optimizer import PricingOptimizer
        optimizer = PricingOptimizer()
        elasticity = optimizer.estimate_elasticity(
            sim_data["orders"], sim_data["order_items"], sim_data["products"],
        )
        assert not elasticity.empty
        assert "estimated_elasticity" in elasticity.columns
        assert "product_sku" in elasticity.columns

    def test_optimize_prices(self, sim_data):
        from sciro.simulation.pricing_optimizer import PricingOptimizer
        optimizer = PricingOptimizer()
        elasticity = optimizer.estimate_elasticity(
            sim_data["orders"], sim_data["order_items"], sim_data["products"],
        )
        results = optimizer.optimize_prices(
            sim_data["products"], elasticity, objective="profit",
        )
        assert not results.empty
        assert "optimal_price" in results.columns
        assert "recommendation" in results.columns

    def test_optimization_summary(self, sim_data):
        from sciro.simulation.pricing_optimizer import PricingOptimizer
        optimizer = PricingOptimizer()
        elasticity = optimizer.estimate_elasticity(
            sim_data["orders"], sim_data["order_items"], sim_data["products"],
        )
        optimizer.optimize_prices(sim_data["products"], elasticity)
        summary = optimizer.get_optimization_summary()
        assert "products_analyzed" in summary
        assert summary["products_analyzed"] > 0


# =============================================================================
# Inventory Optimizer Tests
# =============================================================================
class TestInventoryOptimizer:
    def test_calculate_eoq(self, sim_data):
        from sciro.simulation.inventory_optimizer import InventoryOptimizer
        optimizer = InventoryOptimizer()
        eoq = optimizer.calculate_eoq(
            sim_data["products"], sim_data["orders"], sim_data["order_items"],
        )
        assert not eoq.empty
        assert "eoq" in eoq.columns
        assert "optimal_reorder_point" in eoq.columns
        assert "recommendation" in eoq.columns
        # All EOQs should be non-negative
        assert (eoq["eoq"] >= 0).all()

    def test_optimize_allocation(self, sim_data):
        from sciro.simulation.inventory_optimizer import InventoryOptimizer
        optimizer = InventoryOptimizer()
        alloc = optimizer.optimize_allocation(
            sim_data["inventory"], sim_data["products"],
            sim_data["orders"], sim_data["order_items"],
        )
        assert not alloc.empty
        assert "allocation_status" in alloc.columns
        assert "transfer_recommendation" in alloc.columns

    def test_optimization_summary(self, sim_data):
        from sciro.simulation.inventory_optimizer import InventoryOptimizer
        optimizer = InventoryOptimizer()
        optimizer.calculate_eoq(
            sim_data["products"], sim_data["orders"], sim_data["order_items"],
        )
        optimizer.optimize_allocation(
            sim_data["inventory"], sim_data["products"],
            sim_data["orders"], sim_data["order_items"],
        )
        summary = optimizer.get_optimization_summary()
        assert "eoq" in summary
        assert "allocation" in summary
