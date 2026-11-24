"""
Tests for SCIRO data validators.

Tests Pydantic validation rules for all entity types.
"""

import pytest
from sciro.ingestion.validators import (
    RegionValidator,
    SupplierValidator,
    ProductValidator,
    InventoryValidator,
    ShipmentValidator,
    CustomerValidator,
    OrderValidator,
    OrderItemValidator,
    validate_dataframe,
)
import pandas as pd


# =============================================================================
# Region Validator Tests
# =============================================================================
class TestRegionValidator:
    def test_valid_region(self):
        r = RegionValidator(name="North", country="India")
        assert r.name == "North"
        assert r.country == "India"

    def test_default_country(self):
        r = RegionValidator(name="South")
        assert r.country == "India"

    def test_empty_name_rejected(self):
        with pytest.raises(Exception):
            RegionValidator(name="")

    def test_whitespace_stripped(self):
        r = RegionValidator(name="  North  ")
        assert r.name == "North"


# =============================================================================
# Supplier Validator Tests
# =============================================================================
class TestSupplierValidator:
    def test_valid_supplier(self):
        s = SupplierValidator(
            name="Tata Steel",
            contact_email="contact@tata.com",
            region_name="North",
            lead_time_days=7,
            reliability_score=0.95,
        )
        assert s.name == "Tata Steel"
        assert s.status == "active"

    def test_invalid_email_rejected(self):
        with pytest.raises(Exception):
            SupplierValidator(
                name="Test",
                contact_email="not-an-email",
                region_name="North",
            )

    def test_reliability_out_of_range(self):
        with pytest.raises(Exception):
            SupplierValidator(
                name="Test",
                contact_email="test@test.com",
                region_name="North",
                reliability_score=1.5,
            )

    def test_negative_lead_time_rejected(self):
        with pytest.raises(Exception):
            SupplierValidator(
                name="Test",
                contact_email="test@test.com",
                region_name="North",
                lead_time_days=-1,
            )

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            SupplierValidator(
                name="Test",
                contact_email="test@test.com",
                region_name="North",
                status="unknown",
            )


# =============================================================================
# Product Validator Tests
# =============================================================================
class TestProductValidator:
    def test_valid_product(self):
        p = ProductValidator(
            sku="ELEC-0001",
            name="Widget A",
            category="Electronics",
            unit_price=1000.0,
            cost_price=700.0,
            supplier_name="Tata Steel",
        )
        assert p.sku == "ELEC-0001"
        assert p.status == "active"

    def test_sku_uppercased(self):
        p = ProductValidator(
            sku="elec-0001",
            name="Widget A",
            category="Electronics",
            unit_price=1000.0,
            cost_price=700.0,
            supplier_name="Tata Steel",
        )
        assert p.sku == "ELEC-0001"

    def test_unit_price_below_cost_rejected(self):
        with pytest.raises(Exception):
            ProductValidator(
                sku="ELEC-0001",
                name="Widget A",
                category="Electronics",
                unit_price=500.0,
                cost_price=700.0,
                supplier_name="Tata Steel",
            )

    def test_negative_price_rejected(self):
        with pytest.raises(Exception):
            ProductValidator(
                sku="ELEC-0001",
                name="Widget A",
                category="Electronics",
                unit_price=-100.0,
                cost_price=700.0,
                supplier_name="Tata Steel",
            )


# =============================================================================
# Inventory Validator Tests
# =============================================================================
class TestInventoryValidator:
    def test_valid_inventory(self):
        inv = InventoryValidator(
            product_sku="ELEC-0001",
            region_name="North",
            quantity_on_hand=100,
            quantity_reserved=20,
        )
        assert inv.quantity_on_hand == 100

    def test_reserved_exceeds_onhand_rejected(self):
        with pytest.raises(Exception):
            InventoryValidator(
                product_sku="ELEC-0001",
                region_name="North",
                quantity_on_hand=10,
                quantity_reserved=20,
            )

    def test_negative_quantity_rejected(self):
        with pytest.raises(Exception):
            InventoryValidator(
                product_sku="ELEC-0001",
                region_name="North",
                quantity_on_hand=-5,
            )

    def test_zero_stock_valid(self):
        inv = InventoryValidator(
            product_sku="ELEC-0001",
            region_name="North",
            quantity_on_hand=0,
            quantity_reserved=0,
        )
        assert inv.quantity_on_hand == 0


# =============================================================================
# Customer Validator Tests
# =============================================================================
class TestCustomerValidator:
    def test_valid_customer(self):
        c = CustomerValidator(
            name="Amit Sharma",
            email="amit@company.com",
            region_name="North",
        )
        assert c.tier == "standard"

    def test_invalid_tier_rejected(self):
        with pytest.raises(Exception):
            CustomerValidator(
                name="Test",
                email="test@test.com",
                region_name="North",
                tier="platinum",
            )


# =============================================================================
# Order Validator Tests
# =============================================================================
class TestOrderValidator:
    def test_valid_order(self):
        o = OrderValidator(
            order_number="ORD-202509-00001",
            customer_email="amit@company.com",
            region_name="North",
            order_date="2025-09-01",
            total_amount=5000.0,
        )
        assert o.status == "pending"
        assert o.payment_status == "unpaid"

    def test_negative_total_rejected(self):
        with pytest.raises(Exception):
            OrderValidator(
                order_number="ORD-001",
                customer_email="amit@company.com",
                region_name="North",
                order_date="2025-09-01",
                total_amount=-100.0,
            )


# =============================================================================
# Order Item Validator Tests
# =============================================================================
class TestOrderItemValidator:
    def test_valid_order_item(self):
        oi = OrderItemValidator(
            order_number="ORD-001",
            product_sku="ELEC-0001",
            quantity=5,
            unit_price=1000.0,
            discount=200.0,
        )
        assert oi.quantity == 5

    def test_zero_quantity_rejected(self):
        with pytest.raises(Exception):
            OrderItemValidator(
                order_number="ORD-001",
                product_sku="ELEC-0001",
                quantity=0,
                unit_price=1000.0,
            )

    def test_discount_exceeds_total_rejected(self):
        with pytest.raises(Exception):
            OrderItemValidator(
                order_number="ORD-001",
                product_sku="ELEC-0001",
                quantity=1,
                unit_price=100.0,
                discount=200.0,
            )

    def test_sku_uppercased(self):
        oi = OrderItemValidator(
            order_number="ORD-001",
            product_sku="elec-0001",
            quantity=1,
            unit_price=100.0,
        )
        assert oi.product_sku == "ELEC-0001"


# =============================================================================
# validate_dataframe Tests
# =============================================================================
class TestValidateDataframe:
    def test_valid_regions_df(self):
        df = pd.DataFrame([
            {"name": "North", "country": "India"},
            {"name": "South", "country": "India"},
        ])
        valid, errors = validate_dataframe(df, "regions")
        assert len(valid) == 2
        assert len(errors) == 0

    def test_mixed_valid_invalid(self):
        df = pd.DataFrame([
            {"name": "North", "country": "India"},
            {"name": "", "country": "India"},  # Invalid: empty name
        ])
        valid, errors = validate_dataframe(df, "regions")
        assert len(valid) == 1
        assert len(errors) == 1

    def test_unknown_entity_raises(self):
        df = pd.DataFrame([{"name": "test"}])
        with pytest.raises(ValueError, match="Unknown entity"):
            validate_dataframe(df, "unknown_entity")

    def test_valid_suppliers_df(self):
        df = pd.DataFrame([
            {
                "name": "Tata Steel",
                "contact_email": "contact@tata.com",
                "contact_phone": "+91 9876543210",
                "region_name": "North",
                "lead_time_days": 7,
                "reliability_score": 0.95,
                "status": "active",
            }
        ])
        valid, errors = validate_dataframe(df, "suppliers")
        assert len(valid) == 1
        assert len(errors) == 0
