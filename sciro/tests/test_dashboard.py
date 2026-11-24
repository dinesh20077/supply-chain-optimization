"""
Tests for SCIRO Security, AI Copilot & Dashboard (Module 5).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# Auth Manager Tests
# =============================================================================
class TestAuthManager:
    def test_default_users_created(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        users = auth.list_users()
        usernames = [u["username"] for u in users]
        assert "admin" in usernames
        assert "analyst" in usernames
        assert "viewer" in usernames

    def test_authenticate_valid(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("admin", "admin123")
        assert token is not None

    def test_authenticate_invalid_password(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("admin", "wrongpassword")
        assert token is None

    def test_authenticate_nonexistent_user(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("nonexistent", "password")
        assert token is None

    def test_validate_session(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("admin", "admin123")
        session = auth.validate_session(token)
        assert session is not None
        assert session["role"] == "admin"

    def test_check_permission_admin(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("admin", "admin123")
        assert auth.check_permission(token, "manage_users") is True
        assert auth.check_permission(token, "view_dashboard") is True

    def test_check_permission_viewer(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("viewer", "viewer123")
        assert auth.check_permission(token, "view_dashboard") is True
        assert auth.check_permission(token, "manage_users") is False
        assert auth.check_permission(token, "run_ml_models") is False

    def test_check_permission_analyst(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("analyst", "analyst123")
        assert auth.check_permission(token, "run_ml_models") is True
        assert auth.check_permission(token, "export_data") is True
        assert auth.check_permission(token, "manage_users") is False

    def test_create_user(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        result = auth.create_user("newuser", "pass123", "analyst", "New User")
        assert result is True
        token = auth.authenticate("newuser", "pass123")
        assert token is not None

    def test_create_duplicate_user(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        result = auth.create_user("admin", "admin123", "admin")
        assert result is False

    def test_logout(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("admin", "admin123")
        auth.logout(token)
        session = auth.validate_session(token)
        assert session is None

    def test_deactivate_user(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        token = auth.authenticate("viewer", "viewer123")
        assert token is not None
        auth.deactivate_user("viewer")
        # Session should be invalidated
        session = auth.validate_session(token)
        assert session is None
        # Cannot login again
        token2 = auth.authenticate("viewer", "viewer123")
        assert token2 is None

    def test_update_role(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        result = auth.update_role("viewer", "analyst")
        assert result is True
        token = auth.authenticate("viewer", "viewer123")
        session = auth.validate_session(token)
        assert session["role"] == "analyst"

    def test_get_role_permissions(self):
        from sciro.dashboard.auth import AuthManager
        auth = AuthManager()
        admin_perms = auth.get_role_permissions("admin")
        viewer_perms = auth.get_role_permissions("viewer")
        assert len(admin_perms) > len(viewer_perms)
        assert "manage_users" in admin_perms
        assert "manage_users" not in viewer_perms


# =============================================================================
# AI Copilot Tests
# =============================================================================
@pytest.fixture
def copilot_with_data():
    """Create a copilot with test data loaded."""
    from sciro.dashboard.copilot import AICopilot

    copilot = AICopilot()

    np.random.seed(42)
    base_date = datetime(2026, 1, 1)

    orders_list = []
    items_list = []
    for i in range(60):
        date = base_date + timedelta(days=i)
        order_num = f"ORD-{i:03d}"
        orders_list.append({
            "order_number": order_num,
            "customer_email": f"cust{i%10}@test.com",
            "region_name": "North",
            "order_date": date.strftime("%Y-%m-%d"),
            "status": "delivered",
            "total_amount": 5000.0,
        })
        items_list.append({
            "order_number": order_num,
            "product_sku": "P1",
            "quantity": 5,
            "unit_price": 1000.0,
            "discount": 0,
        })

    copilot._data_cache = {
        "orders": pd.DataFrame(orders_list),
        "order_items": pd.DataFrame(items_list),
        "products": pd.DataFrame([{
            "sku": "P1", "name": "Test Product", "category": "Electronics",
            "unit_price": 1000.0, "cost_price": 600.0,
            "reorder_point": 50, "safety_stock": 20,
            "supplier_name": "Test Supplier",
        }]),
        "suppliers": pd.DataFrame([{
            "name": "Test Supplier", "region_name": "North",
            "status": "active", "lead_time_days": 7,
            "reliability_score": 0.9,
        }]),
        "inventory": pd.DataFrame([{
            "product_sku": "P1", "region_name": "North",
            "quantity_on_hand": 100, "quantity_reserved": 10,
        }]),
        "shipments": pd.DataFrame([{
            "tracking_number": "TRK-001", "supplier_name": "Test Supplier",
            "product_sku": "P1", "destination_region": "North",
            "quantity": 100, "status": "delivered", "shipping_cost": 500,
            "shipped_date": "2026-01-01", "expected_delivery_date": "2026-01-08",
            "actual_delivery_date": "2026-01-07",
        }]),
        "customers": pd.DataFrame([{
            "email": f"cust{i}@test.com", "name": f"Customer {i}",
            "company_name": f"Company {i}", "region_name": "North",
            "tier": np.random.choice(["standard", "gold", "enterprise"]),
        } for i in range(10)]),
        "regions": pd.DataFrame([{"name": "North", "country": "India"}]),
    }

    # Parse dates
    copilot._data_cache["orders"]["order_date"] = pd.to_datetime(
        copilot._data_cache["orders"]["order_date"]
    )

    return copilot


class TestAICopilot:
    def test_intent_classification_revenue(self, copilot_with_data):
        result = copilot_with_data.query("What's our total revenue?")
        assert result["intent"] == "revenue_summary"
        assert result["confidence"] > 0
        assert len(result["response"]) > 0

    def test_intent_classification_inventory(self, copilot_with_data):
        result = copilot_with_data.query("Show me the inventory status")
        assert result["intent"] == "inventory_status"

    def test_intent_classification_supplier_risk(self, copilot_with_data):
        result = copilot_with_data.query("Which suppliers are risky?")
        assert result["intent"] == "supplier_risk"

    def test_intent_classification_help(self, copilot_with_data):
        result = copilot_with_data.query("What can you do?")
        assert result["intent"] == "help"
        assert "SCIRO AI Copilot" in result["response"]

    def test_revenue_summary_response(self, copilot_with_data):
        result = copilot_with_data.query("What's the revenue for the last 30 days?")
        assert "Revenue Summary" in result["response"]
        assert result["data"] is not None

    def test_customer_analysis(self, copilot_with_data):
        result = copilot_with_data.query("Show me customer analysis")
        assert "Customer Analysis" in result["response"]

    def test_unclear_query(self, copilot_with_data):
        result = copilot_with_data.query("xyzzy gibberish")
        assert result["intent"] == "unclear" or result["confidence"] < 0.5

    def test_top_products(self, copilot_with_data):
        result = copilot_with_data.query("What are the top selling products?")
        assert result["intent"] == "top_products"

    def test_no_data_loaded(self):
        from sciro.dashboard.copilot import AICopilot
        copilot = AICopilot()
        result = copilot.query("What's the revenue?")
        assert "No data loaded" in result["response"] or result["intent"] == "help"


# =============================================================================
# Dashboard Data Context & Resilience Tests
# =============================================================================
class TestDataContextResilience:
    def test_ensure_full_data_context_preserves_uploaded_entity(self):
        import streamlit as st
        from sciro.dashboard.app import ensure_full_data_context

        # Simulate user uploading only a custom suppliers dataset
        custom_suppliers = pd.DataFrame([{"name": "Custom Vendor", "status": "active"}])
        st.session_state.data = {"suppliers": custom_suppliers}
        st.session_state.data_loaded = True

        ensure_full_data_context()

        # Check that custom upload was preserved
        assert st.session_state.data["suppliers"].iloc[0]["name"] == "Custom Vendor"
        # Check that missing tables (like orders, products) were backfilled without KeyError
        assert "orders" in st.session_state.data
        assert "products" in st.session_state.data
        assert len(st.session_state.data["orders"]) > 0

    def test_require_datasets_backfills_missing_keys(self):
        import streamlit as st
        from sciro.dashboard.app import require_datasets

        st.session_state.data = {}
        st.session_state.data_loaded = True

        assert require_datasets("orders", "order_items", "products") is True
        assert "orders" in st.session_state.data
        assert len(st.session_state.data["orders"]) > 0
