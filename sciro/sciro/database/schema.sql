-- =============================================================================
-- SCIRO — Database Schema
-- Supply Chain Intelligence & Revenue Optimization Platform
-- =============================================================================
-- This file creates all tables, constraints, and indexes.
-- Designed to be idempotent (safe to run multiple times).
-- =============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS sciro;
SET search_path TO sciro, public;

-- =============================================================================
-- 1. REGIONS
-- Geographic regions for suppliers, customers, and inventory locations
-- =============================================================================
CREATE TABLE IF NOT EXISTS regions (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    country     VARCHAR(100) NOT NULL DEFAULT 'India',
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 2. SUPPLIERS
-- Supplier master data with reliability tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS suppliers (
    id                SERIAL PRIMARY KEY,
    name              VARCHAR(255) NOT NULL UNIQUE,
    contact_email     VARCHAR(255) NOT NULL,
    contact_phone     VARCHAR(50),
    region_id         INTEGER NOT NULL REFERENCES regions(id) ON DELETE RESTRICT,
    lead_time_days    INTEGER NOT NULL DEFAULT 7 CHECK (lead_time_days >= 0),
    reliability_score NUMERIC(3, 2) NOT NULL DEFAULT 1.00
                      CHECK (reliability_score BETWEEN 0.00 AND 1.00),
    status            VARCHAR(20) NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'inactive')),
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_suppliers_region_id ON suppliers(region_id);
CREATE INDEX IF NOT EXISTS idx_suppliers_status ON suppliers(status);

-- =============================================================================
-- 3. PRODUCTS
-- Product catalog with pricing and inventory thresholds
-- =============================================================================
CREATE TABLE IF NOT EXISTS products (
    id                  SERIAL PRIMARY KEY,
    sku                 VARCHAR(50) NOT NULL UNIQUE,
    name                VARCHAR(255) NOT NULL,
    category            VARCHAR(100) NOT NULL,
    unit_price          NUMERIC(12, 2) NOT NULL CHECK (unit_price >= 0),
    cost_price          NUMERIC(12, 2) NOT NULL CHECK (cost_price >= 0),
    weight_kg           NUMERIC(8, 3) CHECK (weight_kg > 0),
    reorder_point       INTEGER NOT NULL DEFAULT 50 CHECK (reorder_point >= 0),
    safety_stock        INTEGER NOT NULL DEFAULT 20 CHECK (safety_stock >= 0),
    primary_supplier_id INTEGER NOT NULL REFERENCES suppliers(id) ON DELETE RESTRICT,
    status              VARCHAR(20) NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'discontinued', 'draft')),
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_product_margin CHECK (unit_price >= cost_price)
);

CREATE INDEX IF NOT EXISTS idx_products_supplier_id ON products(primary_supplier_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_status ON products(status);

-- =============================================================================
-- 4. INVENTORY
-- Stock levels per product per region
-- =============================================================================
CREATE TABLE IF NOT EXISTS inventory (
    id                SERIAL PRIMARY KEY,
    product_id        INTEGER NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    region_id         INTEGER NOT NULL REFERENCES regions(id) ON DELETE RESTRICT,
    quantity_on_hand  INTEGER NOT NULL DEFAULT 0 CHECK (quantity_on_hand >= 0),
    quantity_reserved INTEGER NOT NULL DEFAULT 0 CHECK (quantity_reserved >= 0),
    last_restocked_at TIMESTAMP WITH TIME ZONE,
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_inventory_product_region UNIQUE (product_id, region_id),
    CONSTRAINT chk_reserved_le_onhand CHECK (quantity_reserved <= quantity_on_hand)
);

CREATE INDEX IF NOT EXISTS idx_inventory_product_id ON inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_region_id ON inventory(region_id);

-- =============================================================================
-- 5. SHIPMENTS
-- Inbound shipments from suppliers
-- =============================================================================
CREATE TABLE IF NOT EXISTS shipments (
    id                      SERIAL PRIMARY KEY,
    tracking_number         VARCHAR(100) NOT NULL UNIQUE,
    supplier_id             INTEGER NOT NULL REFERENCES suppliers(id) ON DELETE RESTRICT,
    product_id              INTEGER NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    destination_region_id   INTEGER NOT NULL REFERENCES regions(id) ON DELETE RESTRICT,
    quantity                INTEGER NOT NULL CHECK (quantity > 0),
    status                  VARCHAR(30) NOT NULL DEFAULT 'pending'
                            CHECK (status IN (
                                'pending', 'in_transit', 'delivered',
                                'delayed', 'cancelled'
                            )),
    carrier                 VARCHAR(100),
    shipping_cost           NUMERIC(10, 2) CHECK (shipping_cost >= 0),
    shipped_date            TIMESTAMP WITH TIME ZONE,
    expected_delivery_date  TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_delivery_date    TIMESTAMP WITH TIME ZONE,
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_shipment_dates CHECK (
        actual_delivery_date IS NULL
        OR shipped_date IS NULL
        OR actual_delivery_date >= shipped_date
    )
);

CREATE INDEX IF NOT EXISTS idx_shipments_supplier_id ON shipments(supplier_id);
CREATE INDEX IF NOT EXISTS idx_shipments_product_id ON shipments(product_id);
CREATE INDEX IF NOT EXISTS idx_shipments_region_id ON shipments(destination_region_id);
CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_shipments_dates ON shipments(expected_delivery_date, actual_delivery_date);

-- =============================================================================
-- 6. CUSTOMERS
-- Customer master data
-- =============================================================================
CREATE TABLE IF NOT EXISTS customers (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    company_name  VARCHAR(255),
    region_id     INTEGER NOT NULL REFERENCES regions(id) ON DELETE RESTRICT,
    tier          VARCHAR(20) NOT NULL DEFAULT 'standard'
                  CHECK (tier IN ('standard', 'gold', 'enterprise')),
    created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_region_id ON customers(region_id);
CREATE INDEX IF NOT EXISTS idx_customers_tier ON customers(tier);

-- =============================================================================
-- 7. ORDERS (Sales Order Headers)
-- =============================================================================
CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    order_number    VARCHAR(50) NOT NULL UNIQUE,
    customer_id     INTEGER NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
    region_id       INTEGER NOT NULL REFERENCES regions(id) ON DELETE RESTRICT,
    order_date      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status          VARCHAR(30) NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'confirmed', 'processing',
                        'shipped', 'delivered', 'cancelled'
                    )),
    payment_status  VARCHAR(30) NOT NULL DEFAULT 'unpaid'
                    CHECK (payment_status IN ('unpaid', 'paid', 'refunded')),
    total_amount    NUMERIC(12, 2) NOT NULL DEFAULT 0.00 CHECK (total_amount >= 0),
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_region_id ON orders(region_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

-- =============================================================================
-- 8. ORDER_ITEMS (Sales Order Line Items)
-- =============================================================================
CREATE TABLE IF NOT EXISTS order_items (
    id          SERIAL PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id  INTEGER NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    unit_price  NUMERIC(12, 2) NOT NULL CHECK (unit_price >= 0),
    unit_cost   NUMERIC(12, 2) NOT NULL CHECK (unit_cost >= 0),
    discount    NUMERIC(12, 2) NOT NULL DEFAULT 0.00 CHECK (discount >= 0),
    line_total  NUMERIC(12, 2) NOT NULL CHECK (line_total >= 0),
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_order_product UNIQUE (order_id, product_id)
);

CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);

-- =============================================================================
-- UPDATE TRIGGER FUNCTION
-- Automatically update the updated_at timestamp on row modification
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to all tables with updated_at
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOR tbl IN
        SELECT unnest(ARRAY[
            'regions', 'suppliers', 'products', 'inventory',
            'shipments', 'customers', 'orders'
        ])
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%I_updated_at ON %I; '
            'CREATE TRIGGER trg_%I_updated_at '
            'BEFORE UPDATE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();',
            tbl, tbl, tbl, tbl
        );
    END LOOP;
END;
$$;
