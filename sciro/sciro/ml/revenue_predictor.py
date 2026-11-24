"""
SCIRO Revenue Predictor.

Predicts monthly revenue using gradient boosting trained on
lagged revenue, customer/product metrics, and calendar features.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sciro.ml.feature_engineering import build_revenue_features
from sciro.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_NAME = "revenue_predictor"

FEATURE_COLS = [
    "lag_1_revenue", "lag_2_revenue", "lag_3_revenue",
    "rolling_mean_3_revenue", "revenue_growth_pct",
    "total_orders", "total_units", "avg_order_value",
    "unique_customers", "unique_products",
    "cal_month", "cal_quarter",
]


class RevenuePredictor:
    """Monthly revenue prediction model."""

    def __init__(self, **kwargs):
        defaults = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.params = defaults
        self.model = GradientBoostingRegressor(**defaults)
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def prepare_data(
        self,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build monthly revenue feature matrix."""
        return build_revenue_features(orders_df, order_items_df, products_df)

    def train(
        self,
        features_df: pd.DataFrame,
        target_col: str = "total_revenue",
        test_size: int = 3,
    ) -> Dict[str, float]:
        """Train the revenue prediction model.

        Args:
            features_df: Output of prepare_data().
            target_col: Revenue column to predict.
            test_size: Months to hold out.

        Returns:
            Evaluation metrics dict.
        """
        available = [c for c in FEATURE_COLS if c in features_df.columns]
        df = features_df.replace([np.inf, -np.inf], 0).fillna(0)
        X = df[available]
        y = df[target_col]

        split_idx = len(X) - test_size
        if split_idx < 5:
            self.model.fit(X, y)
            preds = self.model.predict(X)
            y_test = y
        else:
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

        self.metrics = {
            "mae": round(mean_absolute_error(y_test, preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
            "r2": round(r2_score(y_test, preds), 4),
            "mape": round(
                np.mean(np.abs((y_test - preds) / y_test.clip(lower=1))) * 100, 2
            ),
        }
        self.is_fitted = True

        logger.info("RevenuePredictor trained — MAE=₹%.2f, R²=%.4f, MAPE=%.2f%%",
                     self.metrics["mae"], self.metrics["r2"], self.metrics["mape"])
        return self.metrics

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict revenue for given feature rows."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        available = [c for c in FEATURE_COLS if c in features_df.columns]
        df = features_df.copy().replace([np.inf, -np.inf], 0).fillna(0)
        df["predicted_revenue"] = self.model.predict(df[available]).clip(min=0).round(2)
        return df

    def forecast_next_months(
        self,
        features_df: pd.DataFrame,
        months_ahead: int = 3,
    ) -> pd.DataFrame:
        """Forecast revenue for the next N months iteratively.

        Uses the latest month's features and shifts lags forward.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")

        df = features_df.sort_values("month").copy()
        latest = df.iloc[-1].copy()
        forecasts = []

        for i in range(1, months_ahead + 1):
            row = latest.copy()
            # Shift lags
            row["lag_3_revenue"] = row["lag_2_revenue"]
            row["lag_2_revenue"] = row["lag_1_revenue"]
            row["lag_1_revenue"] = row.get("predicted_revenue", row["total_revenue"])
            row["rolling_mean_3_revenue"] = np.mean([
                row["lag_1_revenue"], row["lag_2_revenue"], row["lag_3_revenue"]
            ])

            # Advance calendar
            month_dt = pd.Timestamp(latest["month"]) + pd.DateOffset(months=i)
            row["cal_month"] = month_dt.month
            row["cal_quarter"] = (month_dt.month - 1) // 3 + 1
            row["month"] = month_dt

            available = [c for c in FEATURE_COLS if c in row.index]
            row_df = pd.DataFrame([row])
            row_df = row_df.replace([np.inf, -np.inf], 0).fillna(0)
            pred = self.model.predict(row_df[available])[0]
            row["predicted_revenue"] = max(0, round(pred, 2))
            row["revenue_growth_pct"] = (
                (row["predicted_revenue"] - row["lag_1_revenue"]) / max(row["lag_1_revenue"], 1) * 100
            )

            latest = row.copy()
            forecasts.append(row)

        result = pd.DataFrame(forecasts)
        return result[["month", "predicted_revenue", "cal_month", "cal_quarter"]].reset_index(drop=True)

    def save(self, registry: Optional[ModelRegistry] = None) -> str:
        reg = registry or ModelRegistry()
        return reg.save_model(
            model=self.model, model_name=MODEL_NAME,
            metrics=self.metrics, params=self.params,
            description="Monthly revenue predictor (gradient boosting)",
        )

    def load(self, registry: Optional[ModelRegistry] = None, version: Optional[str] = None):
        reg = registry or ModelRegistry()
        self.model = reg.load_model(MODEL_NAME, version)
        self.is_fitted = True

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        imp = pd.DataFrame({
            "feature": self.model.feature_names_in_,
            "importance": self.model.feature_importances_,
        })
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)
