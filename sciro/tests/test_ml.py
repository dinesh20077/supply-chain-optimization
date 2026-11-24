"""
Tests for SCIRO ML Prediction Engine (Module 3).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# Shared fixture — generates synthetic data matching existing schema
# =============================================================================
@pytest.fixture
def ml_data():
    """Generate realistic test data for ML models."""
    np.random.seed(42)

    # Create 3 months of daily orders
    base_date = datetime(2026, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(90)]

    suppliers = pd.DataFrame([
        {"name": "Supplier A", "region_name": "North", "status": "active",
         "lead_time_days": 10, "reliability_score": 0.65},
        {"name": "Supplier B", "region_name": "South", "status": "active",
         "lead_time_days": 5, "reliability_score": 0.95},
        {"name": "Supplier C", "region_name": "West", "status": "active",
         "lead_time_days": 8, "reliability_score": 0.80},
    ])

    products = pd.DataFrame([
        {"sku": "P1", "name": "Widget A", "category": "Electronics",
         "unit_price": 1000.0, "cost_price": 600.0, "reorder_point": 50,
         "safety_stock": 20, "supplier_name": "Supplier A"},
        {"sku": "P2", "name": "Widget B", "category": "Electronics",
         "unit_price": 500.0, "cost_price": 300.0, "reorder_point": 30,
         "safety_stock": 10, "supplier_name": "Supplier B"},
        {"sku": "P3", "name": "Raw Material X", "category": "Raw Materials",
         "unit_price": 200.0, "cost_price": 100.0, "reorder_point": 100,
         "safety_stock": 40, "supplier_name": "Supplier C"},
    ])

    # Generate orders
    orders_list = []
    items_list = []
    for i, date in enumerate(dates):
        n_orders = np.random.randint(2, 6)
        for j in range(n_orders):
            order_num = f"ORD-{i:03d}-{j:02d}"
            status = np.random.choice(["delivered", "confirmed", "cancelled"], p=[0.7, 0.2, 0.1])
            orders_list.append({
                "order_number": order_num,
                "customer_email": f"customer{np.random.randint(1,20)}@example.com",
                "region_name": np.random.choice(["North", "South", "West"]),
                "order_date": date.strftime("%Y-%m-%d"),
                "status": status,
                "total_amount": np.random.uniform(500, 10000),
            })

            # 1-3 items per order
            n_items = np.random.randint(1, 4)
            used_skus = set()
            for k in range(n_items):
                sku = np.random.choice(["P1", "P2", "P3"])
                if sku in used_skus:
                    continue
                used_skus.add(sku)
                prod = products[products["sku"] == sku].iloc[0]
                qty = np.random.randint(1, 20)
                items_list.append({
                    "order_number": order_num,
                    "product_sku": sku,
                    "quantity": qty,
                    "unit_price": prod["unit_price"],
                    "discount": np.random.choice([0, 50, 100]),
                })

    orders = pd.DataFrame(orders_list)
    order_items = pd.DataFrame(items_list)

    inventory = pd.DataFrame([
        {"product_sku": "P1", "region_name": "North", "quantity_on_hand": 30, "quantity_reserved": 5},
        {"product_sku": "P2", "region_name": "South", "quantity_on_hand": 200, "quantity_reserved": 10},
        {"product_sku": "P3", "region_name": "West", "quantity_on_hand": 0, "quantity_reserved": 0},
        {"product_sku": "P1", "region_name": "South", "quantity_on_hand": 100, "quantity_reserved": 0},
    ])

    shipments = pd.DataFrame([
        {"tracking_number": f"TRK-{i:03d}", "supplier_name": s,
         "product_sku": p, "destination_region": r,
         "quantity": np.random.randint(50, 200),
         "status": np.random.choice(["delivered", "in_transit", "delayed"]),
         "shipping_cost": np.random.uniform(200, 2000),
         "shipped_date": (base_date - timedelta(days=np.random.randint(5, 60))).strftime("%Y-%m-%d"),
         "expected_delivery_date": (base_date - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d"),
         "actual_delivery_date": (base_date - timedelta(days=np.random.randint(-5, 25))).strftime("%Y-%m-%d")
             if np.random.random() > 0.3 else None}
        for i, (s, p, r) in enumerate([
            ("Supplier A", "P1", "North"), ("Supplier A", "P1", "South"),
            ("Supplier B", "P2", "South"), ("Supplier B", "P2", "North"),
            ("Supplier C", "P3", "West"), ("Supplier C", "P3", "South"),
            ("Supplier A", "P1", "West"), ("Supplier B", "P2", "West"),
        ])
    ])

    return {
        "suppliers": suppliers, "products": products, "inventory": inventory,
        "orders": orders, "order_items": order_items, "shipments": shipments,
    }


# =============================================================================
# Demand Forecaster Tests
# =============================================================================
class TestDemandForecaster:
    def test_prepare_data(self, ml_data):
        from sciro.ml.demand_forecaster import DemandForecaster
        forecaster = DemandForecaster()
        features = forecaster.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        assert not features.empty
        assert "lag_1" in features.columns
        assert "rolling_mean_4" in features.columns
        assert "quantity_sold" in features.columns
        assert "category_encoded" in features.columns

    def test_train(self, ml_data):
        from sciro.ml.demand_forecaster import DemandForecaster
        forecaster = DemandForecaster()
        features = forecaster.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        metrics = forecaster.train(features)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert forecaster.is_fitted

    def test_predict(self, ml_data):
        from sciro.ml.demand_forecaster import DemandForecaster
        forecaster = DemandForecaster()
        features = forecaster.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        forecaster.train(features)
        predictions = forecaster.predict(features)
        assert "predicted_demand" in predictions.columns
        assert (predictions["predicted_demand"] >= 0).all()

    def test_forecast_next_period(self, ml_data):
        from sciro.ml.demand_forecaster import DemandForecaster
        forecaster = DemandForecaster()
        features = forecaster.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        forecaster.train(features)
        forecast = forecaster.forecast_next_period(features)
        assert "predicted_demand" in forecast.columns
        assert len(forecast) > 0

    def test_feature_importance(self, ml_data):
        from sciro.ml.demand_forecaster import DemandForecaster
        forecaster = DemandForecaster()
        features = forecaster.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        forecaster.train(features)
        imp = forecaster.get_feature_importance()
        assert not imp.empty
        assert "feature" in imp.columns
        assert "importance" in imp.columns


# =============================================================================
# Revenue Predictor Tests
# =============================================================================
class TestRevenuePredictor:
    def test_prepare_data(self, ml_data):
        from sciro.ml.revenue_predictor import RevenuePredictor
        predictor = RevenuePredictor()
        features = predictor.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        assert not features.empty
        assert "lag_1_revenue" in features.columns
        assert "total_revenue" in features.columns

    def test_train(self, ml_data):
        from sciro.ml.revenue_predictor import RevenuePredictor
        predictor = RevenuePredictor()
        features = predictor.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        metrics = predictor.train(features)
        assert "mae" in metrics
        assert "r2" in metrics
        assert predictor.is_fitted

    def test_forecast(self, ml_data):
        from sciro.ml.revenue_predictor import RevenuePredictor
        predictor = RevenuePredictor()
        features = predictor.prepare_data(
            ml_data["orders"], ml_data["order_items"], ml_data["products"],
        )
        predictor.train(features)
        forecast = predictor.forecast_next_months(features, months_ahead=2)
        assert len(forecast) == 2
        assert "predicted_revenue" in forecast.columns


# =============================================================================
# Stockout Predictor Tests
# =============================================================================
class TestStockoutPredictor:
    def test_prepare_data(self, ml_data):
        from sciro.ml.stockout_predictor import StockoutPredictor
        predictor = StockoutPredictor()
        features = predictor.prepare_data(
            ml_data["inventory"], ml_data["products"], ml_data["orders"],
            ml_data["order_items"], ml_data["suppliers"], ml_data["shipments"],
        )
        assert not features.empty
        assert "is_at_risk" in features.columns
        assert "quantity_available" in features.columns

    def test_train_and_predict(self, ml_data):
        from sciro.ml.stockout_predictor import StockoutPredictor
        predictor = StockoutPredictor()
        features = predictor.prepare_data(
            ml_data["inventory"], ml_data["products"], ml_data["orders"],
            ml_data["order_items"], ml_data["suppliers"], ml_data["shipments"],
        )
        metrics = predictor.train(features)
        assert "accuracy" in metrics
        assert "f1" in metrics

        predictions = predictor.predict(features)
        assert "predicted_at_risk" in predictions.columns
        assert "stockout_risk_proba" in predictions.columns


# =============================================================================
# Supplier Risk Classifier Tests
# =============================================================================
class TestSupplierRiskClassifier:
    def test_prepare_data(self, ml_data):
        from sciro.ml.supplier_risk_classifier import SupplierRiskClassifier
        classifier = SupplierRiskClassifier()
        features = classifier.prepare_data(
            ml_data["suppliers"], ml_data["shipments"], ml_data["products"],
            ml_data["orders"], ml_data["order_items"],
        )
        assert not features.empty
        assert "risk_level" in features.columns
        assert "risk_score" in features.columns

    def test_train_and_predict(self, ml_data):
        from sciro.ml.supplier_risk_classifier import SupplierRiskClassifier
        classifier = SupplierRiskClassifier()
        features = classifier.prepare_data(
            ml_data["suppliers"], ml_data["shipments"], ml_data["products"],
            ml_data["orders"], ml_data["order_items"],
        )
        metrics = classifier.train(features)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics

        predictions = classifier.predict(features)
        assert "predicted_risk_level" in predictions.columns


# =============================================================================
# Model Registry Tests
# =============================================================================
class TestModelRegistry:
    def test_save_and_load(self, tmp_path):
        from sciro.ml.model_registry import ModelRegistry
        from sklearn.linear_model import LinearRegression

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        model = LinearRegression()
        model.fit([[1], [2], [3]], [1, 2, 3])

        path = registry.save_model(
            model, "test_model",
            metrics={"r2": 0.99},
            description="Test model",
        )
        assert path

        loaded = registry.load_model("test_model")
        assert loaded is not None
        assert loaded.predict([[4]])[0] == pytest.approx(4.0, abs=0.1)

    def test_list_models(self, tmp_path):
        from sciro.ml.model_registry import ModelRegistry
        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        registry.save_model("dummy", "model_a", description="Test A")
        registry.save_model("dummy2", "model_b", description="Test B")

        models = registry.list_models()
        assert len(models) == 2

    def test_delete_model(self, tmp_path):
        from sciro.ml.model_registry import ModelRegistry
        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        registry.save_model("dummy", "to_delete")
        registry.delete_model("to_delete")
        models = registry.list_models()
        assert len(models) == 0
