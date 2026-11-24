"""
SCIRO Demand Forecaster.

Predicts future product demand using gradient boosting regression
trained on time-series features (lags, rolling stats, calendar).
"""

import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from sciro.ml.feature_engineering import build_demand_features
from sciro.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_NAME = "demand_forecaster"

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_4",
    "rolling_mean_4", "rolling_std_4",
    "month", "quarter", "week_of_year",
    "is_month_start", "is_month_end",
    "unit_price", "cost_price",
    "category_encoded",
    "order_count",
]


class DemandForecaster:
    """Gradient-boosted demand forecaster per product."""

    def __init__(self, **kwargs):
        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.params = defaults
        self.model = GradientBoostingRegressor(**defaults)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def prepare_data(
        self,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        products_df: pd.DataFrame,
        granularity: str = "weekly",
    ) -> pd.DataFrame:
        """Build feature matrix from raw data."""
        df = build_demand_features(orders_df, order_items_df, products_df, granularity)
        df["category_encoded"] = self.label_encoder.fit_transform(df["category"].astype(str))
        return df

    def train(
        self,
        features_df: pd.DataFrame,
        target_col: str = "quantity_sold",
        test_size: int = 4,
    ) -> Dict[str, float]:
        """Train the demand model using time-series cross-validation.

        Args:
            features_df: Output of prepare_data().
            target_col: Column to predict.
            test_size: Number of periods to hold out for validation.

        Returns:
            Dict of evaluation metrics (MAE, RMSE, R²).
        """
        available = [c for c in FEATURE_COLS if c in features_df.columns]
        df = features_df.replace([np.inf, -np.inf], 0).fillna(0)
        X = df[available]
        y = df[target_col]

        # Time-series split (last test_size periods as test)
        split_idx = len(X) - test_size
        if split_idx < 10:
            # Too few rows for a proper split; train on everything
            self.model.fit(X, y)
            preds = self.model.predict(X)
            X_test, y_test = X, y
        else:
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

        self.metrics = {
            "mae": round(mean_absolute_error(y_test, preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "r2": round(r2_score(y_test, preds), 4),
        }
        self.is_fitted = True

        logger.info("DemandForecaster trained — MAE=%.2f, RMSE=%.2f, R²=%.4f",
                     self.metrics["mae"], self.metrics["rmse"], self.metrics["r2"])
        return self.metrics

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict demand for the given feature rows.

        Args:
            features_df: DataFrame with the same features used in training.

        Returns:
            Input DataFrame with 'predicted_demand' column added.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        available = [c for c in FEATURE_COLS if c in features_df.columns]
        df = features_df.copy().replace([np.inf, -np.inf], 0).fillna(0)
        df["predicted_demand"] = self.model.predict(df[available]).clip(min=0).round(0).astype(int)
        return df

    def forecast_next_period(
        self,
        features_df: pd.DataFrame,
        product_sku: Optional[str] = None,
    ) -> pd.DataFrame:
        """Forecast next period using the most recent features as input.

        For each product, takes the latest row's values and shifts them
        to build a pseudo-forecast input.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        df = features_df.copy()
        if product_sku:
            df = df[df["product_sku"] == product_sku]

        # Use latest row per product as base for next-period features
        latest = df.sort_values("period").groupby("product_sku").last().reset_index()

        # Shift lags
        latest["lag_4"] = latest["lag_2"]
        latest["lag_2"] = latest["lag_1"]
        latest["lag_1"] = latest["quantity_sold"]
        latest["rolling_mean_4"] = (latest["lag_1"] + latest["lag_2"] + latest.get("lag_4", 0)) / 3

        forecast = self.predict(latest)
        return forecast[["product_sku", "category", "predicted_demand"]]

    def save(self, registry: Optional[ModelRegistry] = None) -> str:
        """Save model to the registry."""
        reg = registry or ModelRegistry()
        return reg.save_model(
            model={"model": self.model, "label_encoder": self.label_encoder},
            model_name=MODEL_NAME,
            metrics=self.metrics,
            params=self.params,
            description="Gradient-boosted demand forecaster (per-product, time-series features)",
        )

    def load(self, registry: Optional[ModelRegistry] = None, version: Optional[str] = None):
        """Load model from the registry."""
        reg = registry or ModelRegistry()
        data = reg.load_model(MODEL_NAME, version)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = True
        logger.info("DemandForecaster loaded from registry")

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        available = [c for c in FEATURE_COLS if c in self.model.feature_names_in_]
        imp = pd.DataFrame({
            "feature": available,
            "importance": self.model.feature_importances_,
        })
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
