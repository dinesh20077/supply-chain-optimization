"""
SCIRO Inventory Optimizer.

Optimizes inventory allocation across regions:
1. Economic Order Quantity (EOQ) calculation
2. Safety stock optimization
3. Multi-region inventory redistribution
4. Reorder point optimization

Uses analytical optimization (no LP solver required).
"""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """Inventory allocation and reorder optimization engine."""

    def __init__(self):
        self.eoq_results: Optional[pd.DataFrame] = None
        self.allocation_results: Optional[pd.DataFrame] = None

    def calculate_eoq(
        self,
        products_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        holding_cost_pct: float = 0.20,
        ordering_cost: float = 500.0,
        demand_days: int = 365,
    ) -> pd.DataFrame:
        """Calculate Economic Order Quantity per product.

        EOQ = sqrt(2 * D * S / H)
        Where:
        - D = Annual demand
        - S = Ordering cost per order (₹)
        - H = Annual holding cost per unit (cost_price × holding_cost_pct)

        Also calculates:
        - Optimal reorder point = daily_demand × lead_time + safety_stock
        - Number of orders per year
        - Total inventory cost (ordering + holding)

        Args:
            products_df: Product master data.
            orders_df: Orders for demand estimation.
            order_items_df: Order items for demand.
            holding_cost_pct: Annual holding cost as % of unit cost.
            ordering_cost: Fixed cost per order in ₹.
            demand_days: Days to estimate annual demand.

        Returns:
            DataFrame with EOQ calculations per product.
        """
        from sciro.analytics.inventory import calculate_avg_daily_demand

        demand = calculate_avg_daily_demand(orders_df, order_items_df, days=min(demand_days, 90))

        results = []
        for _, product in products_df.iterrows():
            sku = product["sku"]
            cost_price = float(product.get("cost_price", 0))
            lead_time = int(product.get("lead_time_days", 7)) if "lead_time_days" in product.index else 7

            # Get demand
            product_demand = demand[demand["product_sku"] == sku]
            daily_demand = float(product_demand["avg_daily_demand"].iloc[0]) if not product_demand.empty else 0
            annual_demand = daily_demand * 365

            if annual_demand <= 0 or cost_price <= 0:
                results.append({
                    "product_sku": sku,
                    "product_name": product.get("name", ""),
                    "category": product.get("category", ""),
                    "cost_price": cost_price,
                    "daily_demand": 0,
                    "annual_demand": 0,
                    "eoq": 0,
                    "orders_per_year": 0,
                    "optimal_reorder_point": int(product.get("reorder_point", 0)),
                    "optimal_safety_stock": int(product.get("safety_stock", 0)),
                    "total_annual_cost": 0,
                    "recommendation": "Insufficient demand data.",
                })
                continue

            # Holding cost per unit per year
            H = cost_price * holding_cost_pct
            S = ordering_cost
            D = annual_demand

            # EOQ formula
            eoq = np.sqrt(2 * D * S / H)
            orders_per_year = D / eoq if eoq > 0 else 0

            # Total cost = ordering cost + holding cost
            total_ordering_cost = orders_per_year * S
            total_holding_cost = (eoq / 2) * H
            total_cost = total_ordering_cost + total_holding_cost

            # Optimal safety stock (assume demand std ≈ 30% of mean, 95% service level Z=1.65)
            demand_std = daily_demand * 0.30
            safety_stock = int(np.ceil(1.65 * demand_std * np.sqrt(lead_time)))

            # Reorder point
            reorder_point = int(np.ceil(daily_demand * lead_time + safety_stock))

            results.append({
                "product_sku": sku,
                "product_name": product.get("name", ""),
                "category": product.get("category", ""),
                "cost_price": round(cost_price, 2),
                "daily_demand": round(daily_demand, 2),
                "annual_demand": round(annual_demand, 0),
                "eoq": int(np.ceil(eoq)),
                "orders_per_year": round(orders_per_year, 1),
                "optimal_reorder_point": reorder_point,
                "optimal_safety_stock": safety_stock,
                "current_reorder_point": int(product.get("reorder_point", 0)),
                "current_safety_stock": int(product.get("safety_stock", 0)),
                "total_annual_cost": round(total_cost, 2),
                "recommendation": self._eoq_recommendation(
                    eoq, reorder_point, safety_stock,
                    int(product.get("reorder_point", 0)),
                    int(product.get("safety_stock", 0)),
                ),
            })

        self.eoq_results = pd.DataFrame(results)
        logger.info("Calculated EOQ for %d products", len(self.eoq_results))
        return self.eoq_results

    def optimize_allocation(
        self,
        inventory_df: pd.DataFrame,
        products_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Optimize inventory allocation across regions.

        Identifies overstock and understock situations by comparing
        current stock to demand-proportional ideal allocation.

        Returns:
            DataFrame with rebalancing recommendations per product-region.
        """
        # Regional demand
        orders = orders_df.copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
        valid = orders[orders["status"] != "cancelled"]

        items = order_items_df[
            order_items_df["order_number"].isin(valid["order_number"])
        ].copy()
        items = items.merge(valid[["order_number", "region_name"]], on="order_number", how="left")

        regional_demand = items.groupby(["product_sku", "region_name"])["quantity"].sum().reset_index()
        regional_demand.columns = ["product_sku", "region_name", "total_demand"]

        # Total demand per product
        product_demand = regional_demand.groupby("product_sku")["total_demand"].sum().reset_index()
        product_demand.columns = ["product_sku", "total_product_demand"]

        regional_demand = regional_demand.merge(product_demand, on="product_sku", how="left")
        regional_demand["demand_share"] = (
            regional_demand["total_demand"] / regional_demand["total_product_demand"].clip(lower=1)
        )

        # Current inventory
        inv = inventory_df.copy()
        inv["quantity_available"] = inv["quantity_on_hand"] - inv["quantity_reserved"]

        # Total inventory per product
        product_total_stock = inv.groupby("product_sku")["quantity_available"].sum().reset_index()
        product_total_stock.columns = ["product_sku", "total_stock"]

        # Merge everything
        analysis = inv.merge(regional_demand, on=["product_sku", "region_name"], how="left")
        analysis = analysis.merge(product_total_stock, on="product_sku", how="left")
        analysis = analysis.merge(
            products_df[["sku", "name", "category"]],
            left_on="product_sku", right_on="sku", how="left",
        )

        analysis["demand_share"] = analysis["demand_share"].fillna(0)
        analysis["total_stock"] = analysis["total_stock"].fillna(0)

        # Ideal allocation = total stock × demand share
        analysis["ideal_allocation"] = (analysis["total_stock"] * analysis["demand_share"]).round(0).astype(int)
        analysis["allocation_gap"] = analysis["ideal_allocation"] - analysis["quantity_available"]

        # Recommendation
        analysis["allocation_status"] = np.where(
            analysis["allocation_gap"] > 10, "UNDERSTOCKED",
            np.where(analysis["allocation_gap"] < -10, "OVERSTOCKED", "BALANCED")
        )

        analysis["transfer_recommendation"] = analysis.apply(
            lambda row: (
                f"Transfer {int(abs(row['allocation_gap']))} units TO this region"
                if row["allocation_gap"] > 10
                else (
                    f"Transfer {int(abs(row['allocation_gap']))} units FROM this region"
                    if row["allocation_gap"] < -10
                    else "No transfer needed"
                )
            ),
            axis=1,
        )

        cols = [
            "product_sku", "name", "category", "region_name",
            "quantity_available", "ideal_allocation", "allocation_gap",
            "demand_share", "allocation_status", "transfer_recommendation",
        ]
        available_cols = [c for c in cols if c in analysis.columns]
        self.allocation_results = analysis[available_cols].sort_values(
            "allocation_gap", ascending=True
        ).reset_index(drop=True)

        logger.info(
            "Allocation analysis: %d overstock, %d understock, %d balanced",
            (self.allocation_results["allocation_status"] == "OVERSTOCKED").sum(),
            (self.allocation_results["allocation_status"] == "UNDERSTOCKED").sum(),
            (self.allocation_results["allocation_status"] == "BALANCED").sum(),
        )
        return self.allocation_results

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get combined summary of all optimization results."""
        summary: Dict[str, Any] = {}

        if self.eoq_results is not None and not self.eoq_results.empty:
            eoq = self.eoq_results
            summary["eoq"] = {
                "products_analyzed": len(eoq),
                "avg_eoq": round(eoq["eoq"].mean(), 0),
                "total_annual_cost": round(eoq["total_annual_cost"].sum(), 2),
                "products_needing_reorder_adjustment": int(
                    (eoq["optimal_reorder_point"] != eoq["current_reorder_point"]).sum()
                ),
            }

        if self.allocation_results is not None and not self.allocation_results.empty:
            alloc = self.allocation_results
            summary["allocation"] = {
                "total_records": len(alloc),
                "overstocked": int((alloc["allocation_status"] == "OVERSTOCKED").sum()),
                "understocked": int((alloc["allocation_status"] == "UNDERSTOCKED").sum()),
                "balanced": int((alloc["allocation_status"] == "BALANCED").sum()),
                "top_rebalancing_needs": alloc[
                    alloc["allocation_status"] != "BALANCED"
                ].head(5)[
                    ["product_sku", "region_name", "allocation_gap", "transfer_recommendation"]
                ].to_dict("records"),
            }

        return summary

    def _eoq_recommendation(
        self,
        eoq: float,
        optimal_rop: int,
        optimal_ss: int,
        current_rop: int,
        current_ss: int,
    ) -> str:
        """Generate EOQ-based recommendation."""
        parts = [f"Order {int(eoq)} units per batch."]

        if abs(optimal_rop - current_rop) > 5:
            direction = "increase" if optimal_rop > current_rop else "decrease"
            parts.append(
                f"{direction.capitalize()} reorder point from {current_rop} to {optimal_rop}."
            )

        if abs(optimal_ss - current_ss) > 3:
            direction = "increase" if optimal_ss > current_ss else "decrease"
            parts.append(
                f"{direction.capitalize()} safety stock from {current_ss} to {optimal_ss}."
            )

        if len(parts) == 1:
            parts.append("Current settings are near-optimal.")

        return " ".join(parts)
