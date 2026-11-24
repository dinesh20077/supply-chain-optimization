"""
SCIRO Data Validators.

Pydantic models for validating each entity type before database insertion.
Each model enforces type constraints, value ranges, and business rules.
"""

import re
import logging
from typing import Optional
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Region Validator
# =============================================================================
class RegionValidator(BaseModel):
    """Validates region data."""

    name: str = Field(..., min_length=1, max_length=100)
    country: str = Field(default="India", min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip()


# =============================================================================
# Supplier Validator
# =============================================================================
class SupplierValidator(BaseModel):
    """Validates supplier data."""

    name: str = Field(..., min_length=1, max_length=255)
    contact_email: str = Field(..., max_length=255)
    contact_phone: Optional[str] = Field(default=None, max_length=50)
    region_name: str = Field(..., min_length=1)
    lead_time_days: int = Field(default=7, ge=0)
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = Field(default="active")

    @field_validator("contact_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("active", "inactive"):
            raise ValueError(f"Status must be 'active' or 'inactive', got '{v}'")
        return v

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip()


# =============================================================================
# Product Validator
# =============================================================================
class ProductValidator(BaseModel):
    """Validates product data."""

    sku: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=100)
    unit_price: float = Field(..., ge=0)
    cost_price: float = Field(..., ge=0)
    weight_kg: Optional[float] = Field(default=None, gt=0)
    reorder_point: int = Field(default=50, ge=0)
    safety_stock: int = Field(default=20, ge=0)
    supplier_name: str = Field(..., min_length=1)
    status: str = Field(default="active")

    @model_validator(mode="after")
    def check_margin(self):
        if self.unit_price < self.cost_price:
            raise ValueError(
                f"unit_price ({self.unit_price}) must be >= cost_price ({self.cost_price})"
            )
        return self

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("active", "discontinued", "draft"):
            raise ValueError(f"Status must be 'active', 'discontinued', or 'draft', got '{v}'")
        return v

    @field_validator("sku")
    @classmethod
    def clean_sku(cls, v: str) -> str:
        return v.strip().upper()


# =============================================================================
# Inventory Validator
# =============================================================================
class InventoryValidator(BaseModel):
    """Validates inventory data."""

    product_sku: str = Field(..., min_length=1)
    region_name: str = Field(..., min_length=1)
    quantity_on_hand: int = Field(..., ge=0)
    quantity_reserved: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_reserved(self):
        if self.quantity_reserved > self.quantity_on_hand:
            raise ValueError(
                f"quantity_reserved ({self.quantity_reserved}) cannot exceed "
                f"quantity_on_hand ({self.quantity_on_hand})"
            )
        return self

    @field_validator("product_sku")
    @classmethod
    def clean_sku(cls, v: str) -> str:
        return v.strip().upper()


# =============================================================================
# Shipment Validator
# =============================================================================
class ShipmentValidator(BaseModel):
    """Validates shipment data."""

    tracking_number: str = Field(..., min_length=1, max_length=100)
    supplier_name: str = Field(..., min_length=1)
    product_sku: str = Field(..., min_length=1)
    destination_region: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    status: str = Field(default="pending")
    carrier: Optional[str] = Field(default=None, max_length=100)
    shipping_cost: Optional[float] = Field(default=None, ge=0)
    shipped_date: Optional[str] = None
    expected_delivery_date: str = Field(...)
    actual_delivery_date: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.strip().lower()
        valid = ("pending", "in_transit", "delivered", "delayed", "cancelled")
        if v not in valid:
            raise ValueError(f"Status must be one of {valid}, got '{v}'")
        return v

    @field_validator("shipped_date", "actual_delivery_date", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        """Convert empty strings and NaN to None for optional date fields."""
        if v is None:
            return None
        if isinstance(v, float):
            import math
            return None if math.isnan(v) else str(v)
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("shipping_cost", mode="before")
    @classmethod
    def nan_to_none(cls, v):
        """Convert NaN to None for optional float fields."""
        if v is None:
            return None
        if isinstance(v, float):
            import math
            return None if math.isnan(v) else v
        return v


# =============================================================================
# Customer Validator
# =============================================================================
class CustomerValidator(BaseModel):
    """Validates customer data."""

    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., max_length=255)
    company_name: Optional[str] = Field(default=None, max_length=255)
    region_name: str = Field(..., min_length=1)
    tier: str = Field(default="standard")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("standard", "gold", "enterprise"):
            raise ValueError(f"Tier must be 'standard', 'gold', or 'enterprise', got '{v}'")
        return v


# =============================================================================
# Order Validator
# =============================================================================
class OrderValidator(BaseModel):
    """Validates order header data."""

    order_number: str = Field(..., min_length=1, max_length=50)
    customer_email: str = Field(..., min_length=1)
    region_name: str = Field(..., min_length=1)
    order_date: str = Field(...)
    status: str = Field(default="pending")
    payment_status: str = Field(default="unpaid")
    total_amount: float = Field(..., ge=0)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.strip().lower()
        valid = ("pending", "confirmed", "processing", "shipped", "delivered", "cancelled")
        if v not in valid:
            raise ValueError(f"Status must be one of {valid}, got '{v}'")
        return v

    @field_validator("payment_status")
    @classmethod
    def validate_payment(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("unpaid", "paid", "refunded"):
            raise ValueError(f"payment_status must be 'unpaid', 'paid', or 'refunded', got '{v}'")
        return v


# =============================================================================
# Order Item Validator
# =============================================================================
class OrderItemValidator(BaseModel):
    """Validates order line item data."""

    order_number: str = Field(..., min_length=1)
    product_sku: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    discount: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def check_discount(self):
        max_discount = self.quantity * self.unit_price
        if self.discount > max_discount:
            raise ValueError(
                f"discount ({self.discount}) cannot exceed "
                f"line total ({max_discount})"
            )
        return self

    @field_validator("product_sku")
    @classmethod
    def clean_sku(cls, v: str) -> str:
        return v.strip().upper()


# =============================================================================
# Validation Utility
# =============================================================================
# Map entity names to their validator classes
VALIDATORS = {
    "regions": RegionValidator,
    "suppliers": SupplierValidator,
    "products": ProductValidator,
    "inventory": InventoryValidator,
    "shipments": ShipmentValidator,
    "customers": CustomerValidator,
    "orders": OrderValidator,
    "order_items": OrderItemValidator,
}


def validate_dataframe(
    df, entity_name: str
) -> tuple[list[dict], list[dict]]:
    """Validate a DataFrame row-by-row using the appropriate Pydantic model.

    Args:
        df: pandas DataFrame to validate.
        entity_name: Entity name matching a key in VALIDATORS.

    Returns:
        Tuple of (valid_records, error_records).
        valid_records: list of validated dicts.
        error_records: list of dicts with 'row_index' and 'errors'.
    """
    if entity_name not in VALIDATORS:
        raise ValueError(
            f"Unknown entity '{entity_name}'. Valid: {list(VALIDATORS.keys())}"
        )

    validator_cls = VALIDATORS[entity_name]
    valid_records = []
    error_records = []

    for idx, row in df.iterrows():
        try:
            record = validator_cls(**row.to_dict())
            valid_records.append(record.model_dump())
        except Exception as e:
            error_records.append({
                "row_index": idx,
                "errors": str(e),
                "data": row.to_dict(),
            })

    total = len(valid_records) + len(error_records)
    logger.info(
        "Validation [%s]: %d/%d valid, %d errors",
        entity_name, len(valid_records), total, len(error_records),
    )

    if error_records:
        logger.warning(
            "First validation error [%s]: row %d — %s",
            entity_name, error_records[0]["row_index"], error_records[0]["errors"][:200],
        )

    return valid_records, error_records
