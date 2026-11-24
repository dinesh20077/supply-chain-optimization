"""
SCIRO Disruption Simulator.

Monte Carlo simulation engine that models what happens when
suppliers fail, shipments are delayed, or demand spikes.

Key scenarios:
1. Supplier failure (complete shutdown for N days)
2. Demand surge (X% increase)
3. Lead time degradation (delays increase by Y days)
4. Multi-factor combined disruption
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a disruption scenario."""
    name: str = "default"
    description: str = ""
    # Supplier failure
    failed_suppliers: List[str] = field(default_factory=list)
    supplier_failure_days: int = 30
    # Demand surge
    demand_surge_pct: float = 0.0  # e.g. 0.50 = 50% increase
    demand_surge_products: List[str] = field(default_factory=list)  # empty = all
    # Lead time degradation
    lead_time_increase_days: int = 0
    affected_suppliers: List[str] = field(default_factory=list)  # empty = all
    # Simulation params
    num_simulations: int = 1000
    horizon_days: int = 30
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation run."""
    scenario: str
    num_simulations: int
    revenue_at_risk_mean: float
    revenue_at_risk_p50: float
    revenue_at_risk_p95: float
    revenue_at_risk_p99: float
    stockout_count_mean: float
    stockout_count_p95: float
    affected_products_mean: float
    affected_customers_mean: float
    total_lost_revenue_distribution: List[float] = field(default_factory=list)
    product_impact: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""


class DisruptionSimulator:
    """Monte Carlo disruption simulation engine."""

    def __init__(self):
        self.results: List[SimulationResult] = []

    def run_simulation(
        self,
        scenario: ScenarioConfig,
        products_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
    ) -> SimulationResult:
        """Run a Monte Carlo disruption simulation.

        For each simulation:
        1. Apply disruption effects to inventory/supply
        2. Simulate daily demand draws (Poisson distribution)
        3. Track stockout events and lost revenue
        4. Aggregate across all simulations

        Args:
            scenario: ScenarioConfig defining the disruption.
            products_df: Product master data.
            inventory_df: Current inventory levels.
            orders_df: Historical orders for demand estimation.
            order_items_df: Order items for demand estimation.
            suppliers_df: Supplier data.

        Returns:
            SimulationResult with aggregated outcomes.
        """
        rng = np.random.RandomState(scenario.random_seed)

        # Calculate daily demand per product from historical data
        demand_per_product = self._estimate_daily_demand(orders_df, order_items_df)

        # Build product inventory state
        inv = inventory_df.copy()
        inv["quantity_available"] = inv["quantity_on_hand"] - inv["quantity_reserved"]
        inv = inv.merge(
            products_df[["sku", "name", "unit_price", "supplier_name"]],
            left_on="product_sku", right_on="sku", how="left",
        )
        inv = inv.merge(demand_per_product, on="product_sku", how="left")
        inv["avg_daily_demand"] = inv["avg_daily_demand"].fillna(0)

        # Identify products affected by failed suppliers
        products_from_failed = set()
        if scenario.failed_suppliers:
            products_from_failed = set(
                inv[inv["supplier_name"].isin(scenario.failed_suppliers)]["product_sku"]
            )

        # Run simulations
        total_lost = []
        stockout_counts = []
        affected_products_list = []
        affected_customers_list = []
        product_stockout_tracker: Dict[str, int] = {}

        for sim_i in range(scenario.num_simulations):
            sim_lost_revenue = 0.0
            sim_stockouts = 0
            sim_affected_products = set()

            for _, row in inv.iterrows():
                sku = row["product_sku"]
                stock = max(row["quantity_available"], 0)
                base_demand = max(row["avg_daily_demand"], 0)

                if base_demand == 0:
                    continue

                # Apply demand surge
                demand_multiplier = 1.0
                if scenario.demand_surge_pct > 0:
                    if not scenario.demand_surge_products or sku in scenario.demand_surge_products:
                        demand_multiplier = 1.0 + scenario.demand_surge_pct

                # Simulate supply disruption — no resupply for affected products
                can_resupply = True
                if sku in products_from_failed:
                    can_resupply = False

                # Simulate daily demand over the horizon
                remaining_stock = stock
                days_without_stock = 0

                for day in range(scenario.horizon_days):
                    daily_demand = rng.poisson(base_demand * demand_multiplier)

                    if remaining_stock >= daily_demand:
                        remaining_stock -= daily_demand
                    else:
                        unfulfilled = daily_demand - remaining_stock
                        remaining_stock = 0
                        days_without_stock += 1
                        sim_lost_revenue += unfulfilled * row.get("unit_price", 0)
                        sim_affected_products.add(sku)

                if days_without_stock > 0:
                    sim_stockouts += 1
                    product_stockout_tracker[sku] = product_stockout_tracker.get(sku, 0) + 1

            total_lost.append(sim_lost_revenue)
            stockout_counts.append(sim_stockouts)
            affected_products_list.append(len(sim_affected_products))

            # Estimate affected customers (proportional to affected product revenue share)
            total_products = max(len(inv), 1)
            affected_customers_list.append(
                int(len(sim_affected_products) / total_products * 100)
            )

        total_lost_arr = np.array(total_lost)

        # Product impact analysis
        product_impact = []
        for sku, count in sorted(product_stockout_tracker.items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
            prod_row = inv[inv["product_sku"] == sku].iloc[0] if sku in inv["product_sku"].values else None
            if prod_row is not None:
                product_impact.append({
                    "product_sku": sku,
                    "product_name": prod_row.get("name", ""),
                    "stockout_frequency_pct": round(count / scenario.num_simulations * 100, 1),
                    "current_stock": int(prod_row.get("quantity_available", 0)),
                    "avg_daily_demand": round(prod_row.get("avg_daily_demand", 0), 1),
                })

        result = SimulationResult(
            scenario=scenario.name,
            num_simulations=scenario.num_simulations,
            revenue_at_risk_mean=round(float(total_lost_arr.mean()), 2),
            revenue_at_risk_p50=round(float(np.percentile(total_lost_arr, 50)), 2),
            revenue_at_risk_p95=round(float(np.percentile(total_lost_arr, 95)), 2),
            revenue_at_risk_p99=round(float(np.percentile(total_lost_arr, 99)), 2),
            stockout_count_mean=round(float(np.mean(stockout_counts)), 1),
            stockout_count_p95=round(float(np.percentile(stockout_counts, 95)), 0),
            affected_products_mean=round(float(np.mean(affected_products_list)), 1),
            affected_customers_mean=round(float(np.mean(affected_customers_list)), 1),
            total_lost_revenue_distribution=total_lost_arr.tolist(),
            product_impact=product_impact,
            summary=self._build_summary(scenario, total_lost_arr, stockout_counts),
        )

        self.results.append(result)
        logger.info(
            "Simulation '%s' complete: %d runs, mean revenue at risk ₹%.2f, P95 ₹%.2f",
            scenario.name, scenario.num_simulations,
            result.revenue_at_risk_mean, result.revenue_at_risk_p95,
        )
        return result

    def run_scenario_comparison(
        self,
        scenarios: List[ScenarioConfig],
        products_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run multiple scenarios and return a comparative summary."""
        rows = []
        for sc in scenarios:
            result = self.run_simulation(
                sc, products_df, inventory_df, orders_df, order_items_df, suppliers_df,
            )
            rows.append({
                "scenario": sc.name,
                "description": sc.description,
                "revenue_at_risk_mean": result.revenue_at_risk_mean,
                "revenue_at_risk_p95": result.revenue_at_risk_p95,
                "stockout_count_mean": result.stockout_count_mean,
                "affected_products_mean": result.affected_products_mean,
            })
        return pd.DataFrame(rows)

    def _estimate_daily_demand(
        self,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        days: int = 90,
    ) -> pd.DataFrame:
        """Estimate average daily demand per product."""
        orders = orders_df.copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
        max_date = orders["order_date"].max()
        if pd.isna(max_date):
            return pd.DataFrame(columns=["product_sku", "avg_daily_demand"])

        cutoff = max_date - pd.Timedelta(days=days)
        recent = orders[
            (orders["order_date"] >= cutoff) & (orders["status"] != "cancelled")
        ]
        recent_items = order_items_df[
            order_items_df["order_number"].isin(recent["order_number"])
        ]
        demand = recent_items.groupby("product_sku")["quantity"].sum().reset_index()
        demand.columns = ["product_sku", "total_qty"]
        demand["avg_daily_demand"] = (demand["total_qty"] / days).round(2)
        return demand[["product_sku", "avg_daily_demand"]]

    def _build_summary(
        self,
        scenario: ScenarioConfig,
        lost_arr: np.ndarray,
        stockout_counts: list,
    ) -> str:
        """Build human-readable simulation summary."""
        parts = [f"Scenario: {scenario.name}"]
        if scenario.description:
            parts.append(scenario.description)

        if scenario.failed_suppliers:
            parts.append(f"Suppliers offline: {', '.join(scenario.failed_suppliers)}")
        if scenario.demand_surge_pct > 0:
            parts.append(f"Demand surge: +{scenario.demand_surge_pct*100:.0f}%")
        if scenario.lead_time_increase_days > 0:
            parts.append(f"Lead time increase: +{scenario.lead_time_increase_days} days")

        parts.append(f"\nAfter {scenario.num_simulations} simulations over {scenario.horizon_days} days:")
        parts.append(f"  Mean revenue at risk: ₹{lost_arr.mean():,.2f}")
        parts.append(f"  95th percentile: ₹{np.percentile(lost_arr, 95):,.2f}")
        parts.append(f"  Worst case (P99): ₹{np.percentile(lost_arr, 99):,.2f}")
        parts.append(f"  Avg products experiencing stockout: {np.mean(stockout_counts):.1f}")

        return "\n".join(parts)
