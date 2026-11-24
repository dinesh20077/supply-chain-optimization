"""
SCIRO Inventory Analytics.

Calculates inventory health, turnover, stockout probability,
and estimated stockout dates. Works with DataFrames — no database required.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================
def load_inventory_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSVs needed for inventory analytics.

    Args:
        data_dir: Path to directory containing CSV files.

    Returns:
        Dict with keys: inventory, products, orders, order_items, suppliers, shipments.
    """
    import os

    data = {}
    for name in ("inventory", "products", "orders", "order_items", "suppliers", "shipments"):
        path = os.path.join(data_dir, f"{name}.csv")
        data[name] = pd.read_csv(path)
        logger.info("Loaded %s: %d rows", name, len(data[name]))

    # Parse dates
    data["orders"]["order_date"] = pd.to_datetime(data["orders"]["order_date"], errors="coerce")
    for col in ("shipped_date", "expected_delivery_date", "actual_delivery_date"):
        if col in data["shipments"].columns:
            data["shipments"][col] = pd.to_datetime(data["shipments"][col], errors="coerce")

    return data


# =============================================================================
# Average Daily Demand
# =============================================================================
def calculate_avg_daily_demand(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    days: int = 90,
) -> pd.DataFrame:
    """Calculate average daily demand per product.

    Args:
        orders_df: Orders with order_date (datetime).
        order_items_df: Order line items.
        days: Lookback window in days.

    Returns:
        DataFrame with: product_sku, total_quantity_sold, avg_daily_demand.
    """
    orders = orders_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(orders["order_date"]):
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")

    cutoff = orders["order_date"].max() - pd.Timedelta(days=days)
    recent = orders[
        (orders["order_date"] >= cutoff) & (orders["status"] != "cancelled")
    ]

    recent_items = order_items_df[order_items_df["order_number"].isin(recent["order_number"])]

    demand = recent_items.groupby("product_sku")["quantity"].sum().reset_index()
    demand.columns = ["product_sku", "total_quantity_sold"]
    demand["avg_daily_demand"] = round(demand["total_quantity_sold"] / days, 2)

    logger.info("Calculated demand for %d products over %d days", len(demand), days)
    return demand


# =============================================================================
# Inventory Status / Health
# =============================================================================
def get_inventory_status(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    suppliers_df: Optional[pd.DataFrame] = None,
    demand_days: int = 90,
) -> pd.DataFrame:
    """Calculate comprehensive inventory health per product-region.

    Args:
        inventory_df: Current stock levels.
        products_df: Product master data.
        orders_df: Orders for demand calculation.
        order_items_df: Order items for demand calculation.
        suppliers_df: Optional suppliers data for supplier info.
        demand_days: Lookback window for demand calculation.

    Returns:
        DataFrame with: product_sku, region_name, quantity_on_hand, quantity_reserved,
        quantity_available, reorder_point, safety_stock, avg_daily_demand,
        days_of_supply, stock_status, inventory_value, supplier info.
    """
    inv = inventory_df.copy()
    inv["quantity_available"] = inv["quantity_on_hand"] - inv["quantity_reserved"]

    # Merge product info
    products = products_df[["sku", "name", "category", "unit_price", "cost_price",
                            "reorder_point", "safety_stock", "supplier_name"]].copy()
    inv = inv.merge(products, left_on="product_sku", right_on="sku", how="left")

    # Calculate demand
    demand = calculate_avg_daily_demand(orders_df, order_items_df, days=demand_days)
    inv = inv.merge(demand[["product_sku", "avg_daily_demand"]], on="product_sku", how="left")
    inv["avg_daily_demand"] = inv["avg_daily_demand"].fillna(0)

    # Days of supply
    inv["days_of_supply"] = np.where(
        inv["avg_daily_demand"] > 0,
        (inv["quantity_available"] / inv["avg_daily_demand"]).round(1),
        np.inf,  # No demand = infinite supply
    )

    # Stock status classification
    conditions = [
        inv["quantity_on_hand"] == 0,
        inv["quantity_available"] <= inv["safety_stock"],
        inv["quantity_available"] <= inv["reorder_point"],
    ]
    choices = ["OUT_OF_STOCK", "CRITICAL", "LOW"]
    inv["stock_status"] = np.select(conditions, choices, default="HEALTHY")

    # Inventory valuation
    inv["inventory_value"] = round(inv["quantity_on_hand"] * inv["cost_price"], 2)

    # Suggested reorder quantity
    inv["suggested_reorder_qty"] = np.maximum(
        0,
        (inv["reorder_point"] + inv["safety_stock"]) - inv["quantity_available"],
    ).astype(int)

    # Add supplier info if available
    if suppliers_df is not None:
        supplier_info = suppliers_df[["name", "reliability_score", "lead_time_days"]].copy()
        supplier_info.columns = ["supplier_name", "supplier_reliability", "supplier_lead_time"]
        inv = inv.merge(supplier_info, on="supplier_name", how="left")

    result_cols = [
        "product_sku", "name", "category", "region_name",
        "quantity_on_hand", "quantity_reserved", "quantity_available",
        "reorder_point", "safety_stock", "avg_daily_demand",
        "days_of_supply", "stock_status", "inventory_value",
        "suggested_reorder_qty", "supplier_name", "unit_price", "cost_price",
    ]
    if suppliers_df is not None:
        result_cols.extend(["supplier_reliability", "supplier_lead_time"])

    # Only keep columns that exist
    result_cols = [c for c in result_cols if c in inv.columns]
    result = inv[result_cols].copy()

    logger.info(
        "Inventory status: %d records — %d HEALTHY, %d LOW, %d CRITICAL, %d OUT_OF_STOCK",
        len(result),
        (result["stock_status"] == "HEALTHY").sum(),
        (result["stock_status"] == "LOW").sum(),
        (result["stock_status"] == "CRITICAL").sum(),
        (result["stock_status"] == "OUT_OF_STOCK").sum(),
    )

    return result


# =============================================================================
# Stockout Probability
# =============================================================================
def calculate_stockout_probability(
    inventory_status_df: pd.DataFrame,
    lead_time_buffer_days: int = 5,
) -> pd.DataFrame:
    """Calculate stockout probability per product-region.

    Logic:
    - If already out of stock → probability = 1.0
    - If days_of_supply < supplier_lead_time → high probability
    - Uses: current_stock, daily_demand, lead_time, safety_stock

    Stockout probability = max(0, min(1, 1 - (days_of_supply - lead_time) / lead_time))

    Args:
        inventory_status_df: Output of get_inventory_status().
        lead_time_buffer_days: Additional buffer days for uncertainty.

    Returns:
        DataFrame with added: stockout_probability, stockout_risk, estimated_stockout_date.
    """
    df = inventory_status_df.copy()

    # Effective lead time (supplier lead time + buffer, default 10 if unknown)
    if "supplier_lead_time" in df.columns:
        df["effective_lead_time"] = df["supplier_lead_time"].fillna(10) + lead_time_buffer_days
    else:
        df["effective_lead_time"] = 10 + lead_time_buffer_days

    # Stockout probability calculation
    df["stockout_probability"] = np.where(
        df["stock_status"] == "OUT_OF_STOCK",
        1.0,
        np.where(
            df["avg_daily_demand"] <= 0,
            0.0,  # No demand = no stockout risk
            np.clip(
                1.0 - (df["days_of_supply"] - df["effective_lead_time"]) / df["effective_lead_time"].clip(lower=1),
                0.0,
                1.0,
            ),
        ),
    )
    df["stockout_probability"] = df["stockout_probability"].round(3)

    # Risk classification
    df["stockout_risk"] = pd.cut(
        df["stockout_probability"],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    # Estimated stockout date
    today = pd.Timestamp.now().normalize()
    df["estimated_stockout_date"] = np.where(
        (df["avg_daily_demand"] > 0) & (df["quantity_available"] > 0),
        today + pd.to_timedelta(df["days_of_supply"].clip(upper=365), unit="D"),
        np.where(
            df["stock_status"] == "OUT_OF_STOCK",
            today,
            pd.NaT,
        ),
    )
    df["estimated_stockout_date"] = pd.to_datetime(df["estimated_stockout_date"], errors="coerce")

    # Revenue at risk per product-region (if demand continues and stockout happens)
    df["potential_lost_revenue"] = np.where(
        df["stockout_probability"] > 0.2,
        (df["avg_daily_demand"] * df["effective_lead_time"] * df["unit_price"] * df["stockout_probability"]).round(2),
        0.0,
    )

    logger.info(
        "Stockout risk: %d LOW, %d MEDIUM, %d HIGH, %d CRITICAL",
        (df["stockout_risk"] == "LOW").sum(),
        (df["stockout_risk"] == "MEDIUM").sum(),
        (df["stockout_risk"] == "HIGH").sum(),
        (df["stockout_risk"] == "CRITICAL").sum(),
    )

    return df


# =============================================================================
# Inventory Turnover
# =============================================================================
def calculate_inventory_turnover(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    days: int = 90,
) -> pd.DataFrame:
    """Calculate inventory turnover ratio per product.

    Turnover = COGS (in period) / Average Inventory Value
    Higher turnover = inventory is moving faster.

    Args:
        inventory_df: Current stock levels.
        products_df: Products with cost_price.
        orders_df: Orders for COGS.
        order_items_df: Order items for COGS.
        days: Period for COGS calculation.

    Returns:
        DataFrame with: product_sku, product_name, cogs, avg_inventory_value,
        turnover_ratio, turnover_category.
    """
    orders = orders_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(orders["order_date"]):
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")

    cutoff = orders["order_date"].max() - pd.Timedelta(days=days)
    recent = orders[
        (orders["order_date"] >= cutoff) & (orders["status"] != "cancelled")
    ]

    recent_items = order_items_df[order_items_df["order_number"].isin(recent["order_number"])].copy()

    # COGS per product
    sku_to_cost = products_df.set_index("sku")["cost_price"].to_dict()
    recent_items["cost_price"] = recent_items["product_sku"].map(sku_to_cost)
    recent_items["cogs"] = recent_items["quantity"] * recent_items["cost_price"]

    cogs_by_product = recent_items.groupby("product_sku")["cogs"].sum().reset_index()
    cogs_by_product.columns = ["product_sku", "cogs"]

    # Average inventory value (using current stock as proxy)
    inv_value = inventory_df.copy()
    inv_value = inv_value.merge(
        products_df[["sku", "cost_price", "name"]],
        left_on="product_sku",
        right_on="sku",
        how="left",
    )
    inv_value["inv_value"] = inv_value["quantity_on_hand"] * inv_value["cost_price"]
    avg_inv = inv_value.groupby("product_sku").agg(
        avg_inventory_value=("inv_value", "mean"),
        product_name=("name", "first"),
    ).reset_index()

    # Merge
    result = avg_inv.merge(cogs_by_product, on="product_sku", how="outer")
    result["cogs"] = result["cogs"].fillna(0)
    result["avg_inventory_value"] = result["avg_inventory_value"].fillna(0)

    # Turnover ratio
    result["turnover_ratio"] = np.where(
        result["avg_inventory_value"] > 0,
        (result["cogs"] / result["avg_inventory_value"]).round(2),
        0.0,
    )

    # Annualize
    result["annualized_turnover"] = round(result["turnover_ratio"] * (365 / days), 2)

    # Category
    result["turnover_category"] = pd.cut(
        result["annualized_turnover"],
        bins=[-0.01, 2, 6, 12, float("inf")],
        labels=["SLOW", "MODERATE", "GOOD", "FAST"],
    )

    logger.info("Calculated turnover for %d products", len(result))
    return result.sort_values("turnover_ratio", ascending=False).reset_index(drop=True)


# =============================================================================
# Summary
# =============================================================================
def get_inventory_summary(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    suppliers_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Get high-level inventory KPIs.

    Returns dict with key metrics for the executive dashboard.
    """
    status = get_inventory_status(
        inventory_df, products_df, orders_df, order_items_df, suppliers_df
    )
    stockout = calculate_stockout_probability(status)

    total_value = round(status["inventory_value"].sum(), 2)

    summary = {
        "total_inventory_records": len(status),
        "total_inventory_value": total_value,
        "out_of_stock_count": int((status["stock_status"] == "OUT_OF_STOCK").sum()),
        "critical_stock_count": int((status["stock_status"] == "CRITICAL").sum()),
        "low_stock_count": int((status["stock_status"] == "LOW").sum()),
        "healthy_stock_count": int((status["stock_status"] == "HEALTHY").sum()),
        "high_stockout_risk_count": int(
            stockout["stockout_risk"].isin(["HIGH", "CRITICAL"]).sum()
        ),
        "total_potential_lost_revenue": round(stockout["potential_lost_revenue"].sum(), 2),
        "products_needing_reorder": int((status["suggested_reorder_qty"] > 0).sum()),
        "top_stockout_risks": stockout.nlargest(5, "stockout_probability")[
            ["product_sku", "name", "region_name", "quantity_available",
             "days_of_supply", "stockout_probability", "stockout_risk"]
        ].to_dict("records"),
    }

    logger.info(
        "Inventory summary: ₹%.2f total value, %d high stockout risk",
        total_value, summary["high_stockout_risk_count"],
    )

    return summary
