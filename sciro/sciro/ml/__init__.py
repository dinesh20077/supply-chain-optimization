"""
SCIRO ML Package — AI/ML Prediction Engine (Module 3).

Provides demand forecasting, revenue prediction, stockout prediction,
and supplier risk classification using scikit-learn models.
"""

from sciro.ml.demand_forecaster import DemandForecaster
from sciro.ml.revenue_predictor import RevenuePredictor
from sciro.ml.stockout_predictor import StockoutPredictor
from sciro.ml.supplier_risk_classifier import SupplierRiskClassifier
from sciro.ml.model_registry import ModelRegistry

__all__ = [
    "DemandForecaster",
    "RevenuePredictor",
    "StockoutPredictor",
    "SupplierRiskClassifier",
    "ModelRegistry",
]
