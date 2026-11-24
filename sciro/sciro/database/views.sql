-- =============================================================================
-- SCIRO — Analytics SQL Views
-- Supply Chain Intelligence & Revenue Optimization Platform
-- =============================================================================
-- These views provide pre-computed analytics that Module 2 will consume.
-- All views are created with CREATE OR REPLACE for idempotency.
-- =============================================================================

SET search_path TO sciro, public;

-- =============================================================================
-- 1. SUPPLIER PERFORMANCE VIEW
-- Aggregates shipment data to calculate per-supplier reliability metrics.
-- Used by: Supply Chain Analytics, Supplier Risk Scoring
-- =============================================================================
CREATE OR REPLACE VIEW v_supplier_performance AS
SELECT
    s.id                    AS supplier_id,
    s.name                  AS supplier_name,
    r.name                  AS region_name,
    s.lead_time_days        AS contract_lead_time_days,
    s.reliability_score     AS base_reliability_score,
    s.status                AS supplier_status,

    -- Shipment counts
    COUNT(sh.id)            AS total_shipments,

    COUNT(sh.id) FILTER (WHERE sh.status = 'delivered')
        AS delivered_shipments,

    COUNT(sh.id) FILTER (
        WHERE sh.status = 'delivered'
          AND sh.actual_delivery_date <= sh.expected_delivery_date
    ) AS on_time_shipments,

    COUNT(sh.id) FILTER (
        WHERE sh.status = 'delivered'
          AND sh.actual_delivery_date > sh.expected_delivery_date
    ) AS late_shipments,

    COUNT(sh.id) FILTER (WHERE sh.status = 'delayed')
        AS currently_delayed_shipments,

    COUNT(sh.id) FILTER (WHERE sh.status IN ('pending', 'in_transit'))
        AS in_transit_shipments,

    -- On-time delivery rate (percentage)
    ROUND(
        CASE
            WHEN COUNT(sh.id) FILTER (WHERE sh.status = 'delivered') = 0 THEN NULL
            ELSE (
                COUNT(sh.id) FILTER (
                    WHERE sh.status = 'delivered'
                      AND sh.actual_delivery_date <= sh.expected_delivery_date
                )::NUMERIC
                / COUNT(sh.id) FILTER (WHERE sh.status = 'delivered')
            ) * 100
        END, 2
    ) AS on_time_delivery_pct,

    -- Average actual lead time (days from shipped to delivered)
    ROUND(
        AVG(
            EXTRACT(EPOCH FROM (sh.actual_delivery_date - sh.shipped_date)) / 86400.0
        ) FILTER (WHERE sh.actual_delivery_date IS NOT NULL AND sh.shipped_date IS NOT NULL),
        1
    ) AS avg_actual_lead_time_days,

    -- Lead time variability (std deviation in days)
    ROUND(
        STDDEV(
            EXTRACT(EPOCH FROM (sh.actual_delivery_date - sh.shipped_date)) / 86400.0
        ) FILTER (WHERE sh.actual_delivery_date IS NOT NULL AND sh.shipped_date IS NOT NULL),
        1
    ) AS lead_time_stddev_days,

    -- Average delay (days late for late deliveries only)
    ROUND(
        AVG(
            EXTRACT(EPOCH FROM (sh.actual_delivery_date - sh.expected_delivery_date)) / 86400.0
        ) FILTER (
            WHERE sh.actual_delivery_date > sh.expected_delivery_date
        ),
        1
    ) AS avg_delay_days,

    -- Total shipping cost
    COALESCE(SUM(sh.shipping_cost), 0) AS total_shipping_cost

FROM suppliers s
JOIN regions r ON s.region_id = r.id
LEFT JOIN shipments sh ON s.id = sh.supplier_id
GROUP BY s.id, s.name, r.name, s.lead_time_days, s.reliability_score, s.status;


-- =============================================================================
-- 2. INVENTORY STATUS VIEW
-- Current stock levels with health indicators.
-- Used by: Inventory Analytics, Stockout Prediction
-- =============================================================================
CREATE OR REPLACE VIEW v_inventory_status AS
SELECT
    inv.id              AS inventory_id,
    p.id                AS product_id,
    p.sku,
    p.name              AS product_name,
    p.category,
    r.id                AS region_id,
    r.name              AS region_name,
    inv.quantity_on_hand,
    inv.quantity_reserved,
    (inv.quantity_on_hand - inv.quantity_reserved)  AS quantity_available,
    p.reorder_point,
    p.safety_stock,
    p.unit_price,
    p.cost_price,

    -- Stock health classification
    CASE
        WHEN inv.quantity_on_hand = 0
            THEN 'OUT_OF_STOCK'
        WHEN (inv.quantity_on_hand - inv.quantity_reserved) <= p.safety_stock
            THEN 'CRITICAL'
        WHEN (inv.quantity_on_hand - inv.quantity_reserved) <= p.reorder_point
            THEN 'LOW'
        ELSE 'HEALTHY'
    END AS stock_status,

    -- Inventory valuation
    ROUND(inv.quantity_on_hand * p.cost_price, 2) AS inventory_value,

    -- Suggested reorder quantity
    GREATEST(
        0,
        (p.reorder_point + p.safety_stock) - (inv.quantity_on_hand - inv.quantity_reserved)
    ) AS suggested_reorder_qty,

    -- Supplier info for this product
    s.id                AS supplier_id,
    s.name              AS supplier_name,
    s.reliability_score AS supplier_reliability,
    s.lead_time_days    AS supplier_lead_time,

    inv.last_restocked_at,
    inv.updated_at      AS last_updated

FROM inventory inv
JOIN products p ON inv.product_id = p.id
JOIN regions r ON inv.region_id = r.id
JOIN suppliers s ON p.primary_supplier_id = s.id;


-- =============================================================================
-- 3. REVENUE SUMMARY VIEW
-- Revenue aggregated by product, region, and month.
-- Used by: Business Analytics, Revenue Forecasting
-- =============================================================================
CREATE OR REPLACE VIEW v_revenue_summary AS
SELECT
    r.id                AS region_id,
    r.name              AS region_name,
    p.id                AS product_id,
    p.sku,
    p.name              AS product_name,
    p.category,
    DATE_TRUNC('month', o.order_date)::DATE AS sales_month,

    -- Order metrics
    COUNT(DISTINCT o.id)        AS total_orders,
    SUM(oi.quantity)            AS units_sold,

    -- Revenue metrics
    SUM(oi.quantity * oi.unit_price)     AS gross_revenue,
    SUM(oi.discount)                     AS total_discounts,
    SUM(oi.line_total)                   AS net_revenue,

    -- Cost and profit
    SUM(oi.quantity * oi.unit_cost)      AS total_cogs,
    SUM(oi.line_total) - SUM(oi.quantity * oi.unit_cost) AS gross_profit,

    -- Margin percentage
    ROUND(
        CASE
            WHEN SUM(oi.line_total) = 0 THEN 0
            ELSE (
                (SUM(oi.line_total) - SUM(oi.quantity * oi.unit_cost))
                / SUM(oi.line_total)
            ) * 100
        END, 2
    ) AS gross_margin_pct,

    -- Average order value
    ROUND(
        SUM(oi.line_total) / NULLIF(COUNT(DISTINCT o.id), 0), 2
    ) AS avg_order_value

FROM orders o
JOIN regions r ON o.region_id = r.id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE o.status != 'cancelled'
GROUP BY r.id, r.name, p.id, p.sku, p.name, p.category,
         DATE_TRUNC('month', o.order_date);


-- =============================================================================
-- 4. ORDER SUMMARY VIEW
-- Per-customer order metrics for CLV and segmentation.
-- Used by: Customer Analytics, Churn Prediction
-- =============================================================================
CREATE OR REPLACE VIEW v_order_summary AS
SELECT
    c.id                AS customer_id,
    c.name              AS customer_name,
    c.email             AS customer_email,
    c.company_name,
    c.tier              AS customer_tier,
    r.name              AS region_name,

    -- Order counts
    COUNT(o.id)         AS total_orders,
    COUNT(o.id) FILTER (WHERE o.status = 'delivered')   AS delivered_orders,
    COUNT(o.id) FILTER (WHERE o.status = 'cancelled')   AS cancelled_orders,

    -- Revenue metrics
    COALESCE(SUM(o.total_amount) FILTER (WHERE o.status != 'cancelled'), 0)
        AS total_revenue,

    ROUND(
        AVG(o.total_amount) FILTER (WHERE o.status != 'cancelled'), 2
    ) AS avg_order_value,

    MAX(o.total_amount) FILTER (WHERE o.status != 'cancelled')
        AS max_order_value,

    -- Time metrics
    MIN(o.order_date)   AS first_order_date,
    MAX(o.order_date)   AS last_order_date,

    -- Days since last order (recency)
    EXTRACT(DAY FROM (CURRENT_TIMESTAMP - MAX(o.order_date)))::INTEGER
        AS days_since_last_order,

    -- Payment metrics
    COUNT(o.id) FILTER (WHERE o.payment_status = 'paid')   AS paid_orders,
    COUNT(o.id) FILTER (WHERE o.payment_status = 'unpaid') AS unpaid_orders

FROM customers c
JOIN regions r ON c.region_id = r.id
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email, c.company_name, c.tier, r.name;


-- =============================================================================
-- 5. SHIPMENT ANALYSIS VIEW
-- Shipment delay analysis for supply chain monitoring.
-- Used by: Supply Chain Analytics, Anomaly Detection
-- =============================================================================
CREATE OR REPLACE VIEW v_shipment_analysis AS
SELECT
    sh.id                   AS shipment_id,
    sh.tracking_number,
    s.id                    AS supplier_id,
    s.name                  AS supplier_name,
    p.id                    AS product_id,
    p.sku                   AS product_sku,
    p.name                  AS product_name,
    r.name                  AS destination_region,
    sh.quantity,
    sh.status               AS shipment_status,
    sh.carrier,
    sh.shipping_cost,
    sh.shipped_date,
    sh.expected_delivery_date,
    sh.actual_delivery_date,

    -- Delay calculation (in days)
    CASE
        WHEN sh.actual_delivery_date IS NOT NULL AND sh.expected_delivery_date IS NOT NULL
        THEN EXTRACT(DAY FROM (sh.actual_delivery_date - sh.expected_delivery_date))::INTEGER
        ELSE NULL
    END AS delay_days,

    -- Is delayed flag
    CASE
        WHEN sh.actual_delivery_date > sh.expected_delivery_date THEN TRUE
        WHEN sh.status = 'delayed' THEN TRUE
        ELSE FALSE
    END AS is_delayed,

    -- Expected lead time (shipped to expected delivery)
    CASE
        WHEN sh.shipped_date IS NOT NULL AND sh.expected_delivery_date IS NOT NULL
        THEN EXTRACT(DAY FROM (sh.expected_delivery_date - sh.shipped_date))::INTEGER
        ELSE NULL
    END AS expected_lead_time_days,

    -- Actual lead time (shipped to actual delivery)
    CASE
        WHEN sh.shipped_date IS NOT NULL AND sh.actual_delivery_date IS NOT NULL
        THEN EXTRACT(DAY FROM (sh.actual_delivery_date - sh.shipped_date))::INTEGER
        ELSE NULL
    END AS actual_lead_time_days,

    -- Revenue at risk (value of delayed/pending shipment)
    ROUND(sh.quantity * p.unit_price, 2) AS shipment_value

FROM shipments sh
JOIN suppliers s ON sh.supplier_id = s.id
JOIN products p ON sh.product_id = p.id
JOIN regions r ON sh.destination_region_id = r.id;
