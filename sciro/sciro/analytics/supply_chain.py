"""
SCIRO Supply Chain Analytics.

Calculates supplier reliability metrics, risk scores, shipment analysis,
and supplier dependency. Works with pandas DataFrames — no database required.
"""

import logging
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================
def load_supply_chain_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSVs needed for supply chain analytics.

    Args:
        data_dir: Path to directory containing CSV files.

    Returns:
        Dict with keys: suppliers, shipments, products, regions.
    """
    import os

    data = {}
    for name in ("suppliers", "shipments", "products", "regions"):
        path = os.path.join(data_dir, f"{name}.csv")
        data[name] = pd.read_csv(path)
        logger.info("Loaded %s: %d rows", name, len(data[name]))

    # Parse dates
    for col in ("shipped_date", "expected_delivery_date", "actual_delivery_date"):
        if col in data["shipments"].columns:
            data["shipments"][col] = pd.to_datetime(
                data["shipments"][col], errors="coerce"
            )

    return data


# =============================================================================
# Supplier Performance Metrics
# =============================================================================
def get_supplier_performance(
    suppliers_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate comprehensive performance metrics per supplier.

    Args:
        suppliers_df: Suppliers master data.
        shipments_df: Shipment records with dates parsed as datetime.
        products_df: Products with supplier_name column.

    Returns:
        DataFrame with one row per supplier and all performance metrics.
    """
    # Ensure dates are parsed
    sh = shipments_df.copy()
    for col in ("shipped_date", "expected_delivery_date", "actual_delivery_date"):
        if col in sh.columns and not pd.api.types.is_datetime64_any_dtype(sh[col]):
            sh[col] = pd.to_datetime(sh[col], errors="coerce")

    # Delivered shipments for lead time calculations
    delivered = sh[sh["status"] == "delivered"].copy()
    delivered["actual_lead_time_days"] = (
        (delivered["actual_delivery_date"] - delivered["shipped_date"]).dt.total_seconds() / 86400
    )
    delivered["delay_days"] = (
        (delivered["actual_delivery_date"] - delivered["expected_delivery_date"]).dt.total_seconds() / 86400
    )
    delivered["is_late"] = delivered["delay_days"] > 0

    # Aggregate per supplier
    metrics = []
    for _, supplier in suppliers_df.iterrows():
        name = supplier["name"]
        supplier_shipments = sh[sh["supplier_name"] == name]
        supplier_delivered = delivered[delivered["supplier_name"] == name]

        total = len(supplier_shipments)
        delivered_count = len(supplier_delivered)
        on_time = int((supplier_delivered["is_late"] == False).sum()) if delivered_count > 0 else 0
        late = delivered_count - on_time
        in_transit = int((supplier_shipments["status"].isin(["pending", "in_transit"])).sum())
        delayed_now = int((supplier_shipments["status"] == "delayed").sum())

        # On-time delivery rate
        otd_pct = round((on_time / delivered_count * 100), 2) if delivered_count > 0 else None

        # Lead time stats
        avg_lead = round(supplier_delivered["actual_lead_time_days"].mean(), 1) if delivered_count > 0 else None
        std_lead = round(supplier_delivered["actual_lead_time_days"].std(), 1) if delivered_count > 1 else None

        # Average delay (for late ones only)
        late_shipments = supplier_delivered[supplier_delivered["is_late"]]
        avg_delay = round(late_shipments["delay_days"].mean(), 1) if len(late_shipments) > 0 else 0.0
        max_delay = round(late_shipments["delay_days"].max(), 1) if len(late_shipments) > 0 else 0.0

        # Products supplied
        products_supplied = len(products_df[products_df["supplier_name"] == name])

        # Total shipping cost
        total_cost = round(supplier_shipments["shipping_cost"].sum(), 2)

        metrics.append({
            "supplier_name": name,
            "region_name": supplier.get("region_name", ""),
            "status": supplier.get("status", "active"),
            "contract_lead_time_days": supplier.get("lead_time_days", 0),
            "base_reliability_score": supplier.get("reliability_score", 1.0),
            "total_shipments": total,
            "delivered_shipments": delivered_count,
            "on_time_shipments": on_time,
            "late_shipments": late,
            "in_transit_shipments": in_transit,
            "currently_delayed": delayed_now,
            "on_time_delivery_pct": otd_pct,
            "avg_actual_lead_time_days": avg_lead,
            "lead_time_stddev_days": std_lead,
            "avg_delay_days": avg_delay,
            "max_delay_days": max_delay,
            "products_supplied": products_supplied,
            "total_shipping_cost": total_cost,
        })

    result = pd.DataFrame(metrics)
    logger.info("Calculated performance metrics for %d suppliers", len(result))
    return result


# =============================================================================
# Supplier Risk Scoring
# =============================================================================
def calculate_supplier_risk_scores(
    performance_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calculate composite risk score (0-100) for each supplier.

    Risk is based on:
    - On-time delivery rate (lower = riskier)
    - Average delay days (higher = riskier)
    - Lead time variability (higher = riskier)
    - Current delayed shipments (more = riskier)

    Args:
        performance_df: Output of get_supplier_performance().
        weights: Optional weight overrides. Keys: otd, delay, variability, current_delayed.
                 Defaults: otd=0.40, delay=0.25, variability=0.20, current_delayed=0.15

    Returns:
        Input DataFrame with added columns: risk_score (0-100), risk_level.
    """
    w = weights or {
        "otd": 0.40,
        "delay": 0.25,
        "variability": 0.20,
        "current_delayed": 0.15,
    }

    df = performance_df.copy()

    # Normalize each component to 0-1 scale (higher = riskier)
    # 1. OTD risk: (100 - OTD%) / 100
    df["_otd_risk"] = df["on_time_delivery_pct"].apply(
        lambda x: (100 - x) / 100 if pd.notna(x) else 0.5
    )

    # 2. Delay risk: normalize avg_delay_days to 0-1
    max_delay = df["avg_delay_days"].max()
    if max_delay and max_delay > 0:
        df["_delay_risk"] = df["avg_delay_days"].fillna(0) / max_delay
    else:
        df["_delay_risk"] = 0.0

    # 3. Variability risk: normalize lead_time_stddev
    max_std = df["lead_time_stddev_days"].max()
    if max_std and max_std > 0:
        df["_variability_risk"] = df["lead_time_stddev_days"].fillna(0) / max_std
    else:
        df["_variability_risk"] = 0.0

    # 4. Current delayed risk: any currently delayed shipments
    max_delayed = df["currently_delayed"].max()
    if max_delayed and max_delayed > 0:
        df["_current_risk"] = df["currently_delayed"].fillna(0) / max_delayed
    else:
        df["_current_risk"] = 0.0

    # Composite risk score (0-100)
    df["risk_score"] = (
        w["otd"] * df["_otd_risk"]
        + w["delay"] * df["_delay_risk"]
        + w["variability"] * df["_variability_risk"]
        + w["current_delayed"] * df["_current_risk"]
    ) * 100

    df["risk_score"] = df["risk_score"].round(1).clip(0, 100)

    # Risk level classification
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    # Drop internal columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    logger.info(
        "Risk scores: %d LOW, %d MEDIUM, %d HIGH, %d CRITICAL",
        (df["risk_level"] == "LOW").sum(),
        (df["risk_level"] == "MEDIUM").sum(),
        (df["risk_level"] == "HIGH").sum(),
        (df["risk_level"] == "CRITICAL").sum(),
    )

    return df


# =============================================================================
# Shipment Anomaly Detection
# =============================================================================
def get_shipment_anomalies(
    shipments_df: pd.DataFrame,
    std_threshold: float = 2.0,
) -> pd.DataFrame:
    """Detect anomalous shipments based on delivery delay.

    A shipment is anomalous if its delay exceeds mean + std_threshold × stddev
    for that supplier.

    Args:
        shipments_df: Shipment records with dates.
        std_threshold: Number of standard deviations to flag as anomaly.

    Returns:
        DataFrame of anomalous shipments with delay_days and anomaly_score.
    """
    sh = shipments_df.copy()
    for col in ("shipped_date", "expected_delivery_date", "actual_delivery_date"):
        if col in sh.columns and not pd.api.types.is_datetime64_any_dtype(sh[col]):
            sh[col] = pd.to_datetime(sh[col], errors="coerce")

    delivered = sh[sh["status"] == "delivered"].copy()
    if delivered.empty:
        return pd.DataFrame()

    delivered["delay_days"] = (
        (delivered["actual_delivery_date"] - delivered["expected_delivery_date"]).dt.total_seconds() / 86400
    )

    # Per-supplier mean and stddev of delay
    stats = delivered.groupby("supplier_name")["delay_days"].agg(["mean", "std"]).reset_index()
    stats.columns = ["supplier_name", "supplier_mean_delay", "supplier_std_delay"]
    stats["supplier_std_delay"] = stats["supplier_std_delay"].fillna(0)

    delivered = delivered.merge(stats, on="supplier_name", how="left")
    delivered["anomaly_threshold"] = (
        delivered["supplier_mean_delay"] + std_threshold * delivered["supplier_std_delay"]
    )
    delivered["is_anomaly"] = delivered["delay_days"] > delivered["anomaly_threshold"]
    delivered["anomaly_score"] = np.where(
        delivered["supplier_std_delay"] > 0,
        ((delivered["delay_days"] - delivered["supplier_mean_delay"]) / delivered["supplier_std_delay"]).round(2),
        0.0,
    )

    anomalies = delivered[delivered["is_anomaly"]].copy()
    logger.info("Detected %d anomalous shipments out of %d delivered", len(anomalies), len(delivered))

    return anomalies[
        ["tracking_number", "supplier_name", "product_sku", "destination_region",
         "delay_days", "anomaly_score", "shipped_date", "expected_delivery_date",
         "actual_delivery_date"]
    ].sort_values("anomaly_score", ascending=False)


# =============================================================================
# Supplier Dependency Analysis
# =============================================================================
def get_supplier_dependency(
    products_df: pd.DataFrame,
    order_items_df: Optional[pd.DataFrame] = None,
    orders_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Analyze supplier dependency — concentration risk.

    Args:
        products_df: Products with supplier_name.
        order_items_df: Optional. If provided, calculates revenue-weighted dependency.
        orders_df: Optional. Required with order_items_df for revenue filtering.

    Returns:
        DataFrame with: supplier_name, products_count, product_pct,
        revenue (if order data provided), revenue_pct, dependency_level.
    """
    # Product count dependency
    dep = products_df.groupby("supplier_name").agg(
        products_count=("sku", "count"),
    ).reset_index()

    total_products = len(products_df)
    dep["product_pct"] = round(dep["products_count"] / total_products * 100, 2)

    # Revenue-weighted dependency
    if order_items_df is not None and orders_df is not None:
        # Filter non-cancelled orders
        valid_orders = orders_df[orders_df["status"] != "cancelled"]["order_number"]
        valid_items = order_items_df[order_items_df["order_number"].isin(valid_orders)].copy()
        valid_items["line_total"] = valid_items["quantity"] * valid_items["unit_price"] - valid_items["discount"]

        # Map product_sku to supplier
        sku_to_supplier = products_df.set_index("sku")["supplier_name"].to_dict()
        valid_items["supplier_name"] = valid_items["product_sku"].map(sku_to_supplier)

        revenue_by_supplier = valid_items.groupby("supplier_name")["line_total"].sum().reset_index()
        revenue_by_supplier.columns = ["supplier_name", "revenue"]

        dep = dep.merge(revenue_by_supplier, on="supplier_name", how="left")
        dep["revenue"] = dep["revenue"].fillna(0).round(2)
        total_rev = dep["revenue"].sum()
        dep["revenue_pct"] = round(dep["revenue"] / total_rev * 100, 2) if total_rev > 0 else 0.0
    else:
        dep["revenue"] = None
        dep["revenue_pct"] = None

    # Dependency level
    dep["dependency_level"] = pd.cut(
        dep["product_pct"],
        bins=[-1, 5, 15, 30, 100],
        labels=["LOW", "MODERATE", "HIGH", "CRITICAL"],
    )

    return dep.sort_values("product_pct", ascending=False).reset_index(drop=True)


# =============================================================================
# Summary Function
# =============================================================================
def get_supply_chain_summary(
    suppliers_df: pd.DataFrame,
    shipments_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    order_items_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Get high-level supply chain KPIs.

    Returns dict with:
    - total_suppliers, active_suppliers
    - total_shipments, in_transit, delayed, delivered
    - avg_on_time_delivery_pct
    - high_risk_suppliers (count and list)
    - critical_risk_suppliers (count and list)
    - top_risk_suppliers (top 5 by risk score)
    """
    perf = get_supplier_performance(suppliers_df, shipments_df, products_df)
    scored = calculate_supplier_risk_scores(perf)

    active = suppliers_df[suppliers_df["status"] == "active"]

    sh = shipments_df.copy()
    high_risk = scored[scored["risk_level"].isin(["HIGH", "CRITICAL"])]
    critical_risk = scored[scored["risk_level"] == "CRITICAL"]

    summary = {
        "total_suppliers": len(suppliers_df),
        "active_suppliers": len(active),
        "total_shipments": len(sh),
        "in_transit_shipments": int((sh["status"].isin(["pending", "in_transit"])).sum()),
        "delayed_shipments": int((sh["status"] == "delayed").sum()),
        "delivered_shipments": int((sh["status"] == "delivered").sum()),
        "avg_on_time_delivery_pct": round(scored["on_time_delivery_pct"].mean(), 2),
        "high_risk_supplier_count": len(high_risk),
        "high_risk_suppliers": high_risk["supplier_name"].tolist(),
        "critical_risk_supplier_count": len(critical_risk),
        "critical_risk_suppliers": critical_risk["supplier_name"].tolist(),
        "top_risk_suppliers": scored.nlargest(5, "risk_score")[
            ["supplier_name", "risk_score", "risk_level", "on_time_delivery_pct"]
        ].to_dict("records"),
    }

    logger.info("Supply chain summary: %d high-risk, %d critical suppliers",
                summary["high_risk_supplier_count"], summary["critical_risk_supplier_count"])

    return summary
