"""
SCIRO Supplier Risk Classifier.

Multi-class classifier that predicts supplier risk level
(LOW / MEDIUM / HIGH / CRITICAL) using gradient boosting.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sciro.ml.feature_engineering import build_supplier_risk_features
from sciro.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_NAME = "supplier_risk_classifier"

FEATURE_COLS = [
    "on_time_delivery_pct",
    "avg_delay_days",
    "max_delay_days",
    "lead_time_stddev_days",
    "base_reliability_score",
    "contract_lead_time_days",
    "products_supplied",
    "total_shipping_cost",
    "total_shipments",
    "currently_delayed",
    "revenue_share_pct",
]


class SupplierRiskClassifier:
    """Multi-class supplier risk level prediction."""

    def __init__(self, **kwargs):
        defaults = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.params = defaults
        self.model = GradientBoostingClassifier(**defaults)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def prepare_data(
        self,
        suppliers_df: pd.DataFrame,
        shipments_df: pd.DataFrame,
        products_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build supplier feature matrix with risk labels."""
        return build_supplier_risk_features(
            suppliers_df, shipments_df, products_df, orders_df, order_items_df,
        )

    def train(
        self,
        features_df: pd.DataFrame,
        target_col: str = "risk_level",
    ) -> Dict[str, float]:
        """Train the supplier risk classifier.

        Args:
            features_df: Output of prepare_data().
            target_col: Risk level column.

        Returns:
            Dict of evaluation metrics.
        """
        available = [c for c in FEATURE_COLS if c in features_df.columns]
        df = features_df.copy().replace([np.inf, -np.inf], 0)
        X = df[available].fillna(0)
        y_raw = df[target_col].astype(str)
        y = self.label_encoder.fit_transform(y_raw)

        # Train
        self.model.fit(X, y)
        preds = self.model.predict(X)

        self.metrics = {
            "accuracy": round(accuracy_score(y, preds), 4),
            "f1_weighted": round(f1_score(y, preds, average="weighted", zero_division=0), 4),
        }
        self.is_fitted = True

        logger.info(
            "SupplierRiskClassifier trained — Accuracy=%.4f, F1(weighted)=%.4f",
            self.metrics["accuracy"], self.metrics["f1_weighted"],
        )
        return self.metrics

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict risk level for suppliers.

        Returns:
            DataFrame with 'predicted_risk_level' column.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        available = [c for c in FEATURE_COLS if c in features_df.columns]
        df = features_df.copy().replace([np.inf, -np.inf], 0)
        X = df[available].fillna(0)

        predicted_encoded = self.model.predict(X)
        df["predicted_risk_level"] = self.label_encoder.inverse_transform(predicted_encoded)

        # Prediction probabilities per class
        probas = self.model.predict_proba(X)
        for i, label in enumerate(self.label_encoder.classes_):
            df[f"proba_{label}"] = probas[:, i].round(3)

        return df

    def get_high_risk_suppliers(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Filter suppliers predicted as HIGH or CRITICAL risk."""
        high = predictions_df[
            predictions_df["predicted_risk_level"].isin(["HIGH", "CRITICAL"])
        ].copy()
        cols = ["supplier_name", "predicted_risk_level", "on_time_delivery_pct",
                "avg_delay_days", "products_supplied", "revenue_share_pct"]
        available_cols = [c for c in cols if c in high.columns]
        return high[available_cols].sort_values(
            "predicted_risk_level", ascending=True
        ).reset_index(drop=True)

    def save(self, registry: Optional[ModelRegistry] = None) -> str:
        reg = registry or ModelRegistry()
        return reg.save_model(
            model={"model": self.model, "label_encoder": self.label_encoder},
            model_name=MODEL_NAME,
            metrics=self.metrics,
            params=self.params,
            description="Multi-class supplier risk classifier (GBM)",
        )

    def load(self, registry: Optional[ModelRegistry] = None, version: Optional[str] = None):
        reg = registry or ModelRegistry()
        data = reg.load_model(MODEL_NAME, version)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = True

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        available = [c for c in FEATURE_COLS if c in self.model.feature_names_in_]
        imp = pd.DataFrame({
            "feature": available,
            "importance": self.model.feature_importances_,
        })
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
