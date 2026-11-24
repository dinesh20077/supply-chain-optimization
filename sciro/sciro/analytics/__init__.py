"""
SCIRO Analytics Package.

Exposes core supply chain, inventory, revenue, customer, and
Revenue-at-Risk decision intelligence engines.
"""

from sciro.analytics.supply_chain import (
    get_supplier_performance,
    calculate_supplier_risk_scores,
    get_shipment_anomalies,
    get_supplier_dependency,
    get_supply_chain_summary,
    load_supply_chain_data,
)

from sciro.analytics.inventory import (
    calculate_avg_daily_demand,
    get_inventory_status,
    calculate_stockout_probability,
    calculate_inventory_turnover,
    get_inventory_summary,
    load_inventory_data,
)

from sciro.analytics.revenue import (
    get_revenue_summary,
    get_revenue_growth,
    get_revenue_by_product,
    get_revenue_by_region,
    get_revenue_by_customer,
    get_product_performance,
    get_monthly_revenue_trend,
    load_revenue_data,
)

from sciro.analytics.customer import (
    get_customer_metrics,
    calculate_customer_ltv,
    segment_customers,
    calculate_churn_probability,
    get_customer_summary,
)

from sciro.analytics.risk_engine import (
    calculate_revenue_at_risk_by_supplier,
    get_risk_dashboard_data,
    get_supplier_impact_report,
    explain_risk_chain,
)

__all__ = [
    # Supply chain
    "get_supplier_performance",
    "calculate_supplier_risk_scores",
    "get_shipment_anomalies",
    "get_supplier_dependency",
    "get_supply_chain_summary",
    "load_supply_chain_data",
    # Inventory
    "calculate_avg_daily_demand",
    "get_inventory_status",
    "calculate_stockout_probability",
    "calculate_inventory_turnover",
    "get_inventory_summary",
    "load_inventory_data",
    # Revenue
    "get_revenue_summary",
    "get_revenue_growth",
    "get_revenue_by_product",
    "get_revenue_by_region",
    "get_revenue_by_customer",
    "get_product_performance",
    "get_monthly_revenue_trend",
    "load_revenue_data",
    # Customer
    "get_customer_metrics",
    "calculate_customer_ltv",
    "segment_customers",
    "calculate_churn_probability",
    "get_customer_summary",
    # Risk engine
    "calculate_revenue_at_risk_by_supplier",
    "get_risk_dashboard_data",
    "get_supplier_impact_report",
    "explain_risk_chain",
]
