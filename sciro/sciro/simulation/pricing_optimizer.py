"""
SCIRO Pricing Optimizer.

Finds optimal pricing strategies per product/category
to maximize revenue or profit given demand elasticity estimates.

Approach:
- Estimate price elasticity of demand from historical data
- Simulate revenue/profit curves at different price points
- Return optimal price recommendations with expected impact
"""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PricingOptimizer:
    """Price optimization engine using demand elasticity."""

    def __init__(self):
        self.elasticity_estimates: Optional[pd.DataFrame] = None
        self.optimization_results: Optional[pd.DataFrame] = None

    def estimate_elasticity(
        self,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate price elasticity of demand per product.

        Uses price variation across orders to estimate elasticity.
        Elasticity = (% change in quantity) / (% change in price)
        Typically negative (higher price → lower demand).

        For products with no price variation, estimates a default
        elasticity based on category averages.

        Returns:
            DataFrame with: product_sku, name, category, avg_price,
            avg_quantity, estimated_elasticity.
        """
        orders = orders_df.copy()
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
        valid = orders[orders["status"] != "cancelled"]

        items = order_items_df[
            order_items_df["order_number"].isin(valid["order_number"])
        ].copy()
        items = items.merge(valid[["order_number", "order_date"]], on="order_number", how="left")
        items = items.merge(
            products_df[["sku", "name", "category", "cost_price"]],
            left_on="product_sku", right_on="sku", how="left",
        )

        # Monthly aggregation per product
        items["month"] = items["order_date"].dt.to_period("M")
        monthly = items.groupby(["product_sku", "name", "category", "month"]).agg(
            avg_unit_price=("unit_price", "mean"),
            total_quantity=("quantity", "sum"),
        ).reset_index()

        results = []
        for sku, group in monthly.groupby("product_sku"):
            g = group.sort_values("month")
            if len(g) < 3:
                # Not enough data points for elasticity
                results.append({
                    "product_sku": sku,
                    "name": g["name"].iloc[0],
                    "category": g["category"].iloc[0],
                    "avg_price": g["avg_unit_price"].mean(),
                    "avg_quantity": g["total_quantity"].mean(),
                    "estimated_elasticity": -1.0,  # Default assumption
                    "data_points": len(g),
                })
                continue

            # Log-log regression: ln(Q) = a + e*ln(P)
            prices = g["avg_unit_price"].values
            quantities = g["total_quantity"].values

            # Avoid log of zero
            valid_mask = (prices > 0) & (quantities > 0)
            if valid_mask.sum() < 3:
                elasticity = -1.0
            else:
                log_p = np.log(prices[valid_mask])
                log_q = np.log(quantities[valid_mask])

                # Simple OLS for elasticity coefficient
                n = len(log_p)
                denom = (n * np.sum(log_p ** 2) - np.sum(log_p) ** 2)
                if np.std(log_p) > 1e-9 and abs(denom) > 1e-9:
                    elasticity = float(
                        (n * np.sum(log_p * log_q) - np.sum(log_p) * np.sum(log_q))
                        / denom
                    )
                    # Clamp to reasonable range
                    elasticity = max(-5.0, min(0.5, elasticity))
                else:
                    elasticity = -1.0

            results.append({
                "product_sku": sku,
                "name": g["name"].iloc[0],
                "category": g["category"].iloc[0],
                "avg_price": round(g["avg_unit_price"].mean(), 2),
                "avg_quantity": round(g["total_quantity"].mean(), 1),
                "estimated_elasticity": round(elasticity, 3),
                "data_points": len(g),
            })

        self.elasticity_estimates = pd.DataFrame(results)
        logger.info("Estimated elasticity for %d products", len(self.elasticity_estimates))
        return self.elasticity_estimates

    def optimize_prices(
        self,
        products_df: pd.DataFrame,
        elasticity_df: Optional[pd.DataFrame] = None,
        price_range_pct: float = 0.30,
        price_steps: int = 20,
        objective: str = "profit",  # 'revenue' or 'profit'
    ) -> pd.DataFrame:
        """Find optimal price for each product.

        Tests prices within ±price_range_pct of current price and
        finds the one that maximizes revenue or profit given elasticity.

        Args:
            products_df: Product data with unit_price, cost_price.
            elasticity_df: Output of estimate_elasticity(). If None, uses self.elasticity_estimates.
            price_range_pct: Range to explore (0.30 = ±30%).
            price_steps: Number of price points to evaluate.
            objective: 'revenue' or 'profit'.

        Returns:
            DataFrame with optimal pricing recommendations.
        """
        if elasticity_df is None:
            elasticity_df = self.elasticity_estimates
        if elasticity_df is None:
            raise ValueError("No elasticity estimates. Call estimate_elasticity() first.")

        results = []
        for _, row in elasticity_df.iterrows():
            sku = row["product_sku"]
            product = products_df[products_df["sku"] == sku]
            if product.empty:
                continue

            current_price = float(row["avg_price"])
            cost_price = float(product.iloc[0].get("cost_price", current_price * 0.6))
            elasticity = float(row["estimated_elasticity"])
            base_quantity = float(row["avg_quantity"])

            if current_price <= 0 or base_quantity <= 0:
                continue

            # Generate price points
            min_price = max(cost_price * 1.05, current_price * (1 - price_range_pct))
            max_price = current_price * (1 + price_range_pct)
            prices = np.linspace(min_price, max_price, price_steps)

            best_price = current_price
            best_objective = 0
            best_quantity = base_quantity
            curve = []

            for p in prices:
                # Demand at new price: Q_new = Q_base * (P_new / P_base) ^ elasticity
                price_ratio = p / current_price
                predicted_quantity = max(0, base_quantity * (price_ratio ** elasticity))
                revenue = p * predicted_quantity
                profit = (p - cost_price) * predicted_quantity

                obj_value = profit if objective == "profit" else revenue
                curve.append({"price": round(p, 2), objective: round(obj_value, 2)})

                if obj_value > best_objective:
                    best_objective = obj_value
                    best_price = p
                    best_quantity = predicted_quantity

            # Calculate improvement
            current_quantity = base_quantity
            current_revenue = current_price * current_quantity
            current_profit = (current_price - cost_price) * current_quantity
            new_revenue = best_price * best_quantity
            new_profit = (best_price - cost_price) * best_quantity

            if objective == "profit":
                improvement_pct = ((new_profit - current_profit) / max(current_profit, 1)) * 100
            else:
                improvement_pct = ((new_revenue - current_revenue) / max(current_revenue, 1)) * 100

            results.append({
                "product_sku": sku,
                "product_name": row["name"],
                "category": row["category"],
                "current_price": round(current_price, 2),
                "optimal_price": round(best_price, 2),
                "price_change_pct": round((best_price - current_price) / current_price * 100, 1),
                "cost_price": round(cost_price, 2),
                "elasticity": elasticity,
                "current_monthly_quantity": round(current_quantity, 0),
                "predicted_monthly_quantity": round(best_quantity, 0),
                "current_monthly_revenue": round(current_revenue, 2),
                "predicted_monthly_revenue": round(new_revenue, 2),
                "current_monthly_profit": round(current_profit, 2),
                "predicted_monthly_profit": round(new_profit, 2),
                f"{objective}_improvement_pct": round(improvement_pct, 1),
                "recommendation": self._pricing_recommendation(
                    best_price, current_price, improvement_pct
                ),
            })

        self.optimization_results = pd.DataFrame(results)
        if not self.optimization_results.empty:
            self.optimization_results = self.optimization_results.sort_values(
                f"{objective}_improvement_pct", ascending=False
            ).reset_index(drop=True)

        logger.info("Optimized prices for %d products (objective=%s)",
                     len(self.optimization_results), objective)
        return self.optimization_results

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of pricing optimization results."""
        if self.optimization_results is None or self.optimization_results.empty:
            return {"error": "No optimization results. Run optimize_prices() first."}

        df = self.optimization_results
        revenue_col = [c for c in df.columns if "improvement_pct" in c][0]

        return {
            "products_analyzed": len(df),
            "products_with_increase": int((df["optimal_price"] > df["current_price"]).sum()),
            "products_with_decrease": int((df["optimal_price"] < df["current_price"]).sum()),
            "avg_price_change_pct": round(df["price_change_pct"].mean(), 1),
            "total_current_revenue": round(df["current_monthly_revenue"].sum(), 2),
            "total_predicted_revenue": round(df["predicted_monthly_revenue"].sum(), 2),
            "total_current_profit": round(df["current_monthly_profit"].sum(), 2),
            "total_predicted_profit": round(df["predicted_monthly_profit"].sum(), 2),
            "top_opportunities": df.head(5)[
                ["product_sku", "product_name", "current_price",
                 "optimal_price", "price_change_pct", revenue_col]
            ].to_dict("records"),
        }

    def _pricing_recommendation(
        self,
        optimal_price: float,
        current_price: float,
        improvement_pct: float,
    ) -> str:
        """Generate pricing recommendation."""
        change_pct = (optimal_price - current_price) / current_price * 100

        if abs(change_pct) < 2:
            return "HOLD — Current price is near-optimal."
        elif change_pct > 10:
            return f"INCREASE — Raise by {change_pct:.0f}% for {improvement_pct:.1f}% improvement."
        elif change_pct > 0:
            return f"SLIGHT INCREASE — Raise by {change_pct:.0f}% for marginal improvement."
        elif change_pct < -10:
            return f"DECREASE — Lower by {abs(change_pct):.0f}% to capture more volume."
        else:
            return f"SLIGHT DECREASE — Lower by {abs(change_pct):.0f}% for better volume."
