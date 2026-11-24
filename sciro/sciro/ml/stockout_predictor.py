"""
SCIRO Stockout Predictor.

Binary classifier that predicts which products are at risk of
stockout using random forest trained on inventory, demand, and
supplier features.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from sciro.ml.feature_engineering import build_stockout_features
from sciro.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_NAME = "stockout_predictor"

FEATURE_COLS = [
    "quantity_on_hand", "quantity_reserved", "quantity_available",
    "reorder_point", "safety_stock",
    "avg_daily_demand", "days_of_supply",
    "pending_shipment_qty", "demand_variability",
    "unit_price", "cost_price",
]

# Add supplier features if available
OPTIONAL_FEATURES = ["supplier_reliability", "supplier_lead_time"]


class StockoutPredictor:
    """Random forest stockout risk classifier."""

    def __init__(self, **kwargs):
        defaults = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.params = defaults
        self.model = RandomForestClassifier(**defaults)
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}
        self._feature_cols: list = []

    def prepare_data(
        self,
        inventory_df: pd.DataFrame,
        products_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
        shipments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build stockout feature matrix."""
        return build_stockout_features(
            inventory_df, products_df, orders_df,
            order_items_df, suppliers_df, shipments_df,
        )

    def _get_features(self, df: pd.DataFrame) -> list:
        """Get available feature columns."""
        cols = [c for c in FEATURE_COLS if c in df.columns]
        cols += [c for c in OPTIONAL_FEATURES if c in df.columns]
        return cols

    def train(
        self,
        features_df: pd.DataFrame,
        target_col: str = "is_at_risk",
        cv_folds: int = 3,
    ) -> Dict[str, float]:
        """Train the stockout classifier.

        Args:
            features_df: Output of prepare_data().
            target_col: Binary target column (0/1).
            cv_folds: Cross-validation folds.

        Returns:
            Evaluation metrics dict.
        """
        self._feature_cols = self._get_features(features_df)
        df = features_df.copy()

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        X = df[self._feature_cols].fillna(0)
        y = df[target_col]

        # Ensure enough samples for CV
        n_splits = min(cv_folds, max(2, len(y[y == 1]), len(y[y == 0])))
        if n_splits < 2 or len(y) < 10:
            # Too few samples — just train directly
            self.model.fit(X, y)
            preds = self.model.predict(X)
            y_test = y
        else:
            # Cross-validated training
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1")
            logger.info("CV F1 scores: %s (mean=%.3f)", scores.round(3), scores.mean())

            # Final fit on all data
            self.model.fit(X, y)
            preds = self.model.predict(X)
            y_test = y

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        self.is_fitted = True

        logger.info(
            "StockoutPredictor trained — Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f",
            self.metrics["accuracy"], self.metrics["precision"],
            self.metrics["recall"], self.metrics["f1"],
        )
        return self.metrics

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict stockout risk for inventory rows.

        Returns:
            DataFrame with 'predicted_at_risk' (0/1) and 'stockout_risk_proba' columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        df = features_df.copy().replace([np.inf, -np.inf], 0)
        X = df[self._feature_cols].fillna(0)

        df["predicted_at_risk"] = self.model.predict(X)
        probas = self.model.predict_proba(X)
        # probability of being at-risk (class 1)
        risk_col_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 0
        df["stockout_risk_proba"] = probas[:, risk_col_idx].round(3)

        return df

    def get_at_risk_products(self, predictions_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Filter products predicted as at-risk above a confidence threshold."""
        at_risk = predictions_df[predictions_df["stockout_risk_proba"] >= threshold].copy()
        cols = ["product_sku", "name", "region_name", "quantity_available",
                "avg_daily_demand", "days_of_supply", "stockout_risk_proba",
                "predicted_at_risk"]
        available_cols = [c for c in cols if c in at_risk.columns]
        return at_risk[available_cols].sort_values("stockout_risk_proba", ascending=False).reset_index(drop=True)

    def save(self, registry: Optional[ModelRegistry] = None) -> str:
        reg = registry or ModelRegistry()
        return reg.save_model(
            model={"model": self.model, "feature_cols": self._feature_cols},
            model_name=MODEL_NAME,
            metrics=self.metrics,
            params=self.params,
            description="Random forest stockout risk classifier",
        )

    def load(self, registry: Optional[ModelRegistry] = None, version: Optional[str] = None):
        reg = registry or ModelRegistry()
        data = reg.load_model(MODEL_NAME, version)
        self.model = data["model"]
        self._feature_cols = data["feature_cols"]
        self.is_fitted = True

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        imp = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self.model.feature_importances_,
        })
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
