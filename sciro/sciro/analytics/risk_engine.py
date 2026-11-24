"""
SCIRO Revenue-at-Risk Engine.

The CRITICAL feature of the platform.

Connects the chain:
    Supplier Risk → Inventory Risk → Stockout Probability
    → Orders Affected → Customer Impact → Revenue at Risk

This module doesn't just say "Supplier A is risky." It quantifies:
"Supplier A puts ₹X revenue at risk over the next N days, affecting Y customers
and Z products. Here's why, and here's what you should do."
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sciro.analytics.supply_chain import (
    get_supplier_performance,
    calculate_supplier_risk_scores,
)
from sciro.analytics.inventory import (
    get_inventory_status,
    calculate_stockout_probability,
    calculate_avg_daily_demand,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Revenue-at-Risk: Per Supplier
# =============================================================================
def calculate_revenue_at_risk_by_supplier(
    suppliers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Calculate Revenue-at-Risk for each supplier.

    The chain for each supplier:
    1. Calculate supplier risk score
    2. Find all products supplied by this supplier
    3. For each product, calculate stockout probability
    4. Estimate unfulfilled demand during lead time
    5. Calculate revenue at risk = demand × price × stockout_probability
    6. Identify affected customers

    Args:
        suppliers_df: Supplier master data.
        products_df: Products with supplier_name.
        inventory_df: Current stock levels.
        orders_df: Orders for demand calculation.
        order_items_df: Order items for demand/customer analysis.
        shipments_df: Shipments for supplier performance.
        horizon_days: Risk assessment horizon in days.

    Returns:
        DataFrame with per-supplier risk assessment.
    """
    # Ensure dates are parsed
    orders = orders_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(orders["order_date"]):
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")

    # Step 1: Supplier risk scores
    perf = get_supplier_performance(suppliers_df, shipments_df, products_df)
    scored = calculate_supplier_risk_scores(perf)

    # Step 2: Inventory with stockout probability
    inv_status = get_inventory_status(
        inventory_df, products_df, orders, order_items_df, suppliers_df
    )
    stockout = calculate_stockout_probability(inv_status)

    # Step 3: Demand data
    demand = calculate_avg_daily_demand(orders, order_items_df, days=90)

    # Step 4: Customer impact — who buys each product?
    valid_orders = orders[orders["status"] != "cancelled"]
    valid_items = order_items_df[
        order_items_df["order_number"].isin(valid_orders["order_number"])
    ].copy()
    items_with_customer = valid_items.merge(
        valid_orders[["order_number", "customer_email"]],
        on="order_number",
        how="left",
    )
    customers_per_product = items_with_customer.groupby("product_sku")[
        "customer_email"
    ].nunique().reset_index()
    customers_per_product.columns = ["product_sku", "unique_customers"]

    # Step 5: Build per-supplier risk
    results = []

    for _, supplier_row in scored.iterrows():
        supplier_name = supplier_row["supplier_name"]
        risk_score = supplier_row["risk_score"]
        risk_level = supplier_row["risk_level"]

        # Products from this supplier
        supplier_products = products_df[products_df["supplier_name"] == supplier_name]
        if supplier_products.empty:
            continue

        product_skus = supplier_products["sku"].tolist()

        # Stockout data for supplier's products
        supplier_stockout = stockout[stockout["product_sku"].isin(product_skus)].copy()

        if supplier_stockout.empty:
            results.append({
                "supplier_name": supplier_name,
                "risk_score": risk_score,
                "risk_level": str(risk_level),
                "on_time_delivery_pct": supplier_row.get("on_time_delivery_pct"),
                "avg_delay_days": supplier_row.get("avg_delay_days", 0),
                "products_count": len(supplier_products),
                "products_at_risk": 0,
                "total_revenue_at_risk": 0.0,
                "affected_customers": 0,
                "avg_stockout_probability": 0.0,
                "high_risk_products": [],
                "reasoning": f"Supplier has {len(supplier_products)} products but no inventory data.",
                "recommendation": "No immediate action required.",
            })
            continue

        # Products at high stockout risk
        high_risk_products = supplier_stockout[
            supplier_stockout["stockout_probability"] >= 0.5
        ]

        # Revenue at risk calculation
        # For each product: revenue_at_risk = avg_daily_demand × horizon × unit_price × stockout_prob
        supplier_stockout = supplier_stockout.merge(
            demand[["product_sku", "avg_daily_demand"]],
            on="product_sku",
            how="left",
        )
        # avg_daily_demand already exists from inventory_status; use the freshest
        if "avg_daily_demand_y" in supplier_stockout.columns:
            supplier_stockout["avg_daily_demand"] = supplier_stockout["avg_daily_demand_y"].fillna(
                supplier_stockout.get("avg_daily_demand_x", 0)
            )
            supplier_stockout.drop(
                columns=[c for c in supplier_stockout.columns if c.endswith("_x") or c.endswith("_y")],
                errors="ignore",
                inplace=True,
            )

        supplier_stockout["revenue_at_risk"] = (
            supplier_stockout["avg_daily_demand"].fillna(0)
            * horizon_days
            * supplier_stockout["unit_price"].fillna(0)
            * supplier_stockout["stockout_probability"].fillna(0)
        ).round(2)

        total_revenue_at_risk = round(supplier_stockout["revenue_at_risk"].sum(), 2)

        # Affected customers
        customer_impact = supplier_stockout.merge(
            customers_per_product, on="product_sku", how="left"
        )
        affected_customers = int(customer_impact["unique_customers"].fillna(0).sum())

        # Build high-risk product details
        top_risk_products = supplier_stockout.nlargest(5, "revenue_at_risk")
        high_risk_product_list = []
        for _, p in top_risk_products.iterrows():
            if p.get("revenue_at_risk", 0) > 0:
                high_risk_product_list.append({
                    "product_sku": p.get("product_sku", ""),
                    "product_name": p.get("name", ""),
                    "stock_available": int(p.get("quantity_available", 0)),
                    "daily_demand": round(p.get("avg_daily_demand", 0), 1),
                    "days_of_supply": round(p.get("days_of_supply", 0), 1),
                    "stockout_probability": round(p.get("stockout_probability", 0), 3),
                    "revenue_at_risk": round(p.get("revenue_at_risk", 0), 2),
                })

        # Build reasoning
        reasoning = _build_reasoning(supplier_row, supplier_stockout, high_risk_products)
        recommendation = _build_recommendation(risk_level, supplier_row, high_risk_products)

        results.append({
            "supplier_name": supplier_name,
            "risk_score": risk_score,
            "risk_level": str(risk_level),
            "on_time_delivery_pct": supplier_row.get("on_time_delivery_pct"),
            "avg_delay_days": supplier_row.get("avg_delay_days", 0),
            "products_count": len(supplier_products),
            "products_at_risk": len(high_risk_products),
            "total_revenue_at_risk": total_revenue_at_risk,
            "affected_customers": affected_customers,
            "avg_stockout_probability": round(
                supplier_stockout["stockout_probability"].mean(), 3
            ),
            "high_risk_products": high_risk_product_list,
            "reasoning": reasoning,
            "recommendation": recommendation,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("total_revenue_at_risk", ascending=False).reset_index(drop=True)

    logger.info(
        "Revenue-at-Risk: Total ₹%.2f across %d suppliers, %d with HIGH+ risk",
        result_df["total_revenue_at_risk"].sum(),
        len(result_df),
        (result_df["risk_level"].isin(["HIGH", "CRITICAL"])).sum(),
    )

    return result_df


# =============================================================================
# Reasoning and Recommendations
# =============================================================================
def _build_reasoning(
    supplier_row: pd.Series,
    stockout_df: pd.DataFrame,
    high_risk_products: pd.DataFrame,
) -> str:
    """Build human-readable reasoning for the risk assessment."""
    parts = []

    otd = supplier_row.get("on_time_delivery_pct")
    if otd is not None:
        parts.append(f"On-time delivery rate is {otd}%.")

    avg_delay = supplier_row.get("avg_delay_days", 0)
    if avg_delay and avg_delay > 0:
        parts.append(f"Average delivery delay is {avg_delay} days.")

    lead_std = supplier_row.get("lead_time_stddev_days")
    if lead_std is not None and lead_std > 3:
        parts.append(f"Lead time variability is high ({lead_std} days std dev).")

    n_risk = len(high_risk_products)
    if n_risk > 0:
        parts.append(
            f"{n_risk} product(s) have >50% stockout probability."
        )

    out_of_stock = (stockout_df.get("stock_status", pd.Series()) == "OUT_OF_STOCK").sum()
    if out_of_stock > 0:
        parts.append(f"{int(out_of_stock)} product(s) are already out of stock.")

    return " ".join(parts) if parts else "No significant risk factors identified."


def _build_recommendation(
    risk_level: str,
    supplier_row: pd.Series,
    high_risk_products: pd.DataFrame,
) -> str:
    """Build actionable recommendation based on risk level."""
    risk_str = str(risk_level)

    if risk_str == "CRITICAL":
        return (
            "URGENT: Activate alternative suppliers immediately. "
            "Place emergency restock orders for affected products. "
            "Notify customers of potential delays."
        )
    elif risk_str == "HIGH":
        return (
            "Contact supplier to address delivery reliability. "
            "Increase safety stock for affected products. "
            "Identify and qualify backup suppliers."
        )
    elif risk_str == "MEDIUM":
        return (
            "Monitor supplier performance closely. "
            "Review reorder points for supplied products. "
            "Begin evaluating alternative suppliers as contingency."
        )
    else:
        return "Continue routine monitoring. No immediate action required."


# =============================================================================
# Aggregated Risk Dashboard Data
# =============================================================================
def get_risk_dashboard_data(
    suppliers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Get aggregated risk data for the executive dashboard.

    Returns:
        Dict with:
        - total_revenue_at_risk
        - high_risk_supplier_count
        - products_at_stockout_risk
        - affected_customers
        - risk_by_supplier (list of top risks)
        - risk_trend_data (for charting)
    """
    risk_df = calculate_revenue_at_risk_by_supplier(
        suppliers_df, products_df, inventory_df,
        orders_df, order_items_df, shipments_df,
    )

    total_rar = round(risk_df["total_revenue_at_risk"].sum(), 2)

    dashboard = {
        "total_revenue_at_risk": total_rar,
        "high_risk_supplier_count": int(
            risk_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()
        ),
        "critical_supplier_count": int(
            (risk_df["risk_level"] == "CRITICAL").sum()
        ),
        "total_products_at_risk": int(risk_df["products_at_risk"].sum()),
        "total_affected_customers": int(risk_df["affected_customers"].sum()),
        "top_risk_suppliers": risk_df.head(5)[
            ["supplier_name", "risk_score", "risk_level",
             "total_revenue_at_risk", "products_at_risk",
             "affected_customers", "recommendation"]
        ].to_dict("records"),
        "risk_distribution": {
            "LOW": int((risk_df["risk_level"] == "LOW").sum()),
            "MEDIUM": int((risk_df["risk_level"] == "MEDIUM").sum()),
            "HIGH": int((risk_df["risk_level"] == "HIGH").sum()),
            "CRITICAL": int((risk_df["risk_level"] == "CRITICAL").sum()),
        },
    }

    logger.info(
        "Risk dashboard: ₹%.2f total revenue at risk, %d high-risk suppliers",
        total_rar, dashboard["high_risk_supplier_count"],
    )

    return dashboard


# =============================================================================
# Detailed Supplier Impact Report
# =============================================================================
def get_supplier_impact_report(
    supplier_name: str,
    suppliers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
    horizon_days: int = 30,
) -> Dict[str, Any]:
    """Get detailed impact report for a specific supplier.

    This is what the AI Copilot would return when asked:
    "What happens if Supplier X stops deliveries?"

    Args:
        supplier_name: Name of the supplier to analyze.
        Other args: DataFrames for calculation.
        horizon_days: Analysis horizon.

    Returns:
        Detailed dict with full risk chain explanation.
    """
    # Get full risk data
    risk_df = calculate_revenue_at_risk_by_supplier(
        suppliers_df, products_df, inventory_df,
        orders_df, order_items_df, shipments_df,
        horizon_days=horizon_days,
    )

    supplier_risk = risk_df[risk_df["supplier_name"] == supplier_name]

    if supplier_risk.empty:
        return {
            "error": f"Supplier '{supplier_name}' not found.",
            "available_suppliers": risk_df["supplier_name"].tolist(),
        }

    row = supplier_risk.iloc[0]

    # Get supplier's products detail
    supplier_products = products_df[products_df["supplier_name"] == supplier_name]

    # Get recent orders for these products
    orders_copy = orders_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(orders_copy["order_date"]):
        orders_copy["order_date"] = pd.to_datetime(orders_copy["order_date"], errors="coerce")

    valid_orders = orders_copy[orders_copy["status"] != "cancelled"]
    valid_items = order_items_df[
        order_items_df["order_number"].isin(valid_orders["order_number"])
    ]
    supplier_items = valid_items[
        valid_items["product_sku"].isin(supplier_products["sku"])
    ]

    # Recent revenue from this supplier's products
    supplier_items_with_total = supplier_items.copy()
    supplier_items_with_total["line_total"] = (
        supplier_items_with_total["quantity"]
        * supplier_items_with_total["unit_price"]
        - supplier_items_with_total["discount"]
    )
    recent_revenue = round(supplier_items_with_total["line_total"].sum(), 2)

    report = {
        "supplier_name": supplier_name,
        "risk_score": row["risk_score"],
        "risk_level": row["risk_level"],
        "on_time_delivery_pct": row["on_time_delivery_pct"],
        "avg_delay_days": row["avg_delay_days"],
        "products_supplied": row["products_count"],
        "products_at_risk": row["products_at_risk"],
        "revenue_at_risk": row["total_revenue_at_risk"],
        "affected_customers": row["affected_customers"],
        "historical_revenue_from_supplier": recent_revenue,
        "high_risk_products": row["high_risk_products"],
        "risk_chain": {
            "step_1_supplier_risk": f"Risk score: {row['risk_score']}/100 ({row['risk_level']})",
            "step_2_products_affected": f"{row['products_count']} products supplied, {row['products_at_risk']} at stockout risk",
            "step_3_inventory_impact": f"Average stockout probability: {row['avg_stockout_probability']:.1%}",
            "step_4_revenue_impact": f"₹{row['total_revenue_at_risk']:,.2f} revenue at risk over {horizon_days} days",
            "step_5_customer_impact": f"{row['affected_customers']} customers potentially affected",
        },
        "reasoning": row["reasoning"],
        "recommendation": row["recommendation"],
        "analysis_horizon_days": horizon_days,
        "analysis_timestamp": datetime.now().isoformat(),
    }

    return report


# =============================================================================
# Human-Readable Risk Explanation
# =============================================================================
def explain_risk_chain(
    supplier_name: str,
    risk_data: pd.DataFrame,
) -> str:
    """Generate a human-readable explanation of the risk chain.

    Args:
        supplier_name: Supplier to explain.
        risk_data: Output of calculate_revenue_at_risk_by_supplier().

    Returns:
        Multi-line string explaining the risk chain.
    """
    row = risk_data[risk_data["supplier_name"] == supplier_name]

    if row.empty:
        return f"No risk data found for supplier '{supplier_name}'."

    r = row.iloc[0]

    lines = [
        f"=== Risk Analysis: {supplier_name} ===",
        "",
        f"Risk Score: {r['risk_score']}/100 ({r['risk_level']})",
        f"On-Time Delivery: {r['on_time_delivery_pct']}%",
        f"Average Delay: {r['avg_delay_days']} days",
        "",
        f"Products supplied: {r['products_count']}",
        f"Products at stockout risk: {r['products_at_risk']}",
        f"Revenue at Risk: Rs {r['total_revenue_at_risk']:,.2f}",
        f"Affected Customers: {r['affected_customers']}",
        "",
        "--- Risk Chain ---",
        f"1. Supplier has {r['on_time_delivery_pct']}% on-time delivery rate",
        f"2. This creates delivery uncertainty for {r['products_count']} products",
        f"3. {r['products_at_risk']} products face high stockout probability",
        f"4. Estimated Rs {r['total_revenue_at_risk']:,.2f} revenue could be lost",
        f"5. {r['affected_customers']} customers may be affected",
        "",
        "--- Reasoning ---",
        r["reasoning"],
        "",
        "--- Recommendation ---",
        r["recommendation"],
    ]

    return "\n".join(lines)
