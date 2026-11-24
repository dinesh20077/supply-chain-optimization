"""
SCIRO Feature Engineering.

Shared feature-building utilities used by all ML models.
Extracts time-series, categorical, and aggregation features from raw DataFrames.
"""

import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Time-series features for orders/demand
# =============================================================================
def build_demand_features(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    granularity: str = "weekly",
) -> pd.DataFrame:
    """Build demand feature matrix per product per time period.

    Features per period:
    - quantity_sold, revenue, order_count
    - lag_1, lag_2, lag_4 (lagged quantities)
    - rolling_mean_4, rolling_std_4 (4-period rolling stats)
    - month, week_of_year, quarter, is_month_start, is_month_end
    - category (encoded later), unit_price, cost_price

    Args:
        orders_df: Orders with order_date.
        order_items_df: Order line items.
        products_df: Product master data.
        granularity: 'weekly' or 'monthly'.

    Returns:
        DataFrame with one row per product per time period.
    """
    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    valid = orders[orders["status"] != "cancelled"]

    items = order_items_df[
        order_items_df["order_number"].isin(valid["order_number"])
    ].copy()
    items = items.merge(valid[["order_number", "order_date"]], on="order_number", how="left")
    items = items.merge(
        products_df[["sku", "category", "unit_price", "cost_price"]],
        left_on="product_sku",
        right_on="sku",
        how="left",
        suffixes=("", "_product"),
    )
    if "unit_price" not in items.columns and "unit_price_product" in items.columns:
        items["unit_price"] = items["unit_price_product"]

    # Period column
    if granularity == "weekly":
        items["period"] = items["order_date"].dt.to_period("W").apply(lambda x: x.start_time)
    else:
        items["period"] = items["order_date"].dt.to_period("M").apply(lambda x: x.start_time)

    # Aggregate per product-period
    agg = items.groupby(["product_sku", "period", "category", "unit_price", "cost_price"]).agg(
        quantity_sold=("quantity", "sum"),
        revenue=("quantity", lambda x: (x * items.loc[x.index, "unit_price"]).sum()),
        order_count=("order_number", "nunique"),
    ).reset_index()

    agg = agg.sort_values(["product_sku", "period"]).reset_index(drop=True)

    # Lag features per product
    features = []
    for sku, group in agg.groupby("product_sku"):
        g = group.copy().sort_values("period")
        g["lag_1"] = g["quantity_sold"].shift(1)
        g["lag_2"] = g["quantity_sold"].shift(2)
        g["lag_4"] = g["quantity_sold"].shift(4)
        g["rolling_mean_4"] = g["quantity_sold"].rolling(4, min_periods=1).mean()
        g["rolling_std_4"] = g["quantity_sold"].rolling(4, min_periods=1).std().fillna(0)
        features.append(g)

    result = pd.concat(features, ignore_index=True)

    # Calendar features
    result["month"] = result["period"].dt.month
    result["quarter"] = result["period"].dt.quarter
    result["week_of_year"] = result["period"].dt.isocalendar().week.astype(int)
    result["is_month_start"] = result["period"].dt.is_month_start.astype(int)
    result["is_month_end"] = result["period"].dt.is_month_end.astype(int)

    result = result.dropna(subset=["lag_1"]).reset_index(drop=True)

    logger.info("Built demand features: %d rows for %d products", len(result), result["product_sku"].nunique())
    return result


# =============================================================================
# Revenue features (monthly aggregation)
# =============================================================================
def build_revenue_features(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    regions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build monthly revenue feature matrix.

    Features per month:
    - total_revenue, total_orders, total_units, avg_order_value
    - unique_customers, unique_products
    - lag_1_revenue, lag_2_revenue, lag_3_revenue
    - rolling_mean_3_revenue, revenue_growth_pct
    - month, quarter

    Returns:
        DataFrame with one row per month.
    """
    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    valid = orders[orders["status"] != "cancelled"]

    items = order_items_df[
        order_items_df["order_number"].isin(valid["order_number"])
    ].copy()
    items["discount"] = items["discount"].fillna(0)
    items["line_total"] = items["quantity"] * items["unit_price"] - items["discount"]
    items = items.merge(valid[["order_number", "order_date", "customer_email"]], on="order_number", how="left")

    items["month"] = items["order_date"].dt.to_period("M").apply(lambda x: x.start_time)

    monthly = items.groupby("month").agg(
        total_revenue=("line_total", "sum"),
        total_orders=("order_number", "nunique"),
        total_units=("quantity", "sum"),
        unique_customers=("customer_email", "nunique"),
        unique_products=("product_sku", "nunique"),
    ).reset_index().sort_values("month")

    monthly["avg_order_value"] = monthly["total_revenue"] / monthly["total_orders"].clip(lower=1)

    # Lag features
    monthly["lag_1_revenue"] = monthly["total_revenue"].shift(1)
    monthly["lag_2_revenue"] = monthly["total_revenue"].shift(2)
    monthly["lag_3_revenue"] = monthly["total_revenue"].shift(3)
    monthly["rolling_mean_3_revenue"] = monthly["total_revenue"].rolling(3, min_periods=1).mean()
    monthly["revenue_growth_pct"] = monthly["total_revenue"].pct_change() * 100

    # Calendar
    monthly["cal_month"] = monthly["month"].dt.month
    monthly["cal_quarter"] = monthly["month"].dt.quarter

    monthly = monthly.dropna(subset=["lag_1_revenue"]).reset_index(drop=True)

    logger.info("Built revenue features: %d monthly rows", len(monthly))
    return monthly


# =============================================================================
# Inventory / stockout features
# =============================================================================
def build_stockout_features(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for stockout prediction.

    Features per product-region:
    - quantity_on_hand, quantity_reserved, quantity_available
    - reorder_point, safety_stock
    - avg_daily_demand, days_of_supply
    - supplier_reliability, supplier_lead_time
    - pending_shipment_qty
    - demand_variability (std of daily demand)
    - unit_price, cost_price

    Target: stock_status mapped to binary (1=at_risk if CRITICAL or OUT_OF_STOCK)

    Returns:
        DataFrame ready for classification.
    """
    from sciro.analytics.inventory import get_inventory_status

    status = get_inventory_status(
        inventory_df, products_df, orders_df, order_items_df, suppliers_df
    )

    # Pending shipment quantities
    pending = shipments_df[shipments_df["status"].isin(["pending", "in_transit"])]
    pending_qty = pending.groupby("product_sku")["quantity"].sum().reset_index()
    pending_qty.columns = ["product_sku", "pending_shipment_qty"]

    result = status.merge(pending_qty, on="product_sku", how="left")
    result["pending_shipment_qty"] = result["pending_shipment_qty"].fillna(0)

    # Binary target
    result["is_at_risk"] = result["stock_status"].isin(["CRITICAL", "OUT_OF_STOCK"]).astype(int)

    # Demand variability from order items
    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    valid = orders[orders["status"] != "cancelled"]
    items = order_items_df[order_items_df["order_number"].isin(valid["order_number"])].copy()
    items = items.merge(valid[["order_number", "order_date"]], on="order_number", how="left")
    items["week"] = items["order_date"].dt.to_period("W")

    weekly_demand = items.groupby(["product_sku", "week"])["quantity"].sum().reset_index()
    demand_std = weekly_demand.groupby("product_sku")["quantity"].std().fillna(0).reset_index()
    demand_std.columns = ["product_sku", "demand_variability"]

    result = result.merge(demand_std, on="product_sku", how="left")
    result["demand_variability"] = result["demand_variability"].fillna(0)

    logger.info("Built stockout features: %d rows, %d at-risk", len(result), result["is_at_risk"].sum())
    return result


# =============================================================================
# Supplier risk features
# =============================================================================
def build_supplier_risk_features(
    suppliers_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build supplier feature matrix for risk classification.

    Features per supplier:
    - on_time_delivery_pct, avg_delay_days, max_delay_days
    - lead_time_stddev_days, base_reliability_score
    - products_supplied, total_shipping_cost
    - revenue_share_pct (% of total revenue from this supplier's products)

    Target: risk_level from calculate_supplier_risk_scores.

    Returns:
        DataFrame with supplier features and risk labels.
    """
    from sciro.analytics.supply_chain import (
        get_supplier_performance,
        calculate_supplier_risk_scores,
    )

    perf = get_supplier_performance(suppliers_df, shipments_df, products_df)
    scored = calculate_supplier_risk_scores(perf)

    # Revenue share per supplier
    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    valid = orders[orders["status"] != "cancelled"]
    items = order_items_df[order_items_df["order_number"].isin(valid["order_number"])].copy()
    items["line_total"] = items["quantity"] * items["unit_price"] - items["discount"].fillna(0)

    sku_to_supplier = products_df.set_index("sku")["supplier_name"].to_dict()
    items["supplier_name"] = items["product_sku"].map(sku_to_supplier)

    rev_by_supplier = items.groupby("supplier_name")["line_total"].sum().reset_index()
    rev_by_supplier.columns = ["supplier_name", "supplier_revenue"]
    total_rev = rev_by_supplier["supplier_revenue"].sum()
    rev_by_supplier["revenue_share_pct"] = (
        rev_by_supplier["supplier_revenue"] / total_rev * 100 if total_rev > 0 else 0
    )

    result = scored.merge(rev_by_supplier, on="supplier_name", how="left")
    result["supplier_revenue"] = result["supplier_revenue"].fillna(0)
    result["revenue_share_pct"] = result["revenue_share_pct"].fillna(0)

    logger.info("Built supplier risk features: %d suppliers", len(result))
    return result
