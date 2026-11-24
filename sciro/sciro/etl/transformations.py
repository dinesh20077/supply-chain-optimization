import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def add_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add derived date features: year, month, quarter, day_of_week, is_weekend."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")
        
    result = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
        try:
            result[date_column] = pd.to_datetime(result[date_column])
        except Exception as e:
            logger.error(f"Could not convert column '{date_column}' to datetime: {e}")
            raise ValueError(f"Could not convert column '{date_column}' to datetime: {e}")
            
    result[f'{date_column}_year'] = result[date_column].dt.year
    result[f'{date_column}_month'] = result[date_column].dt.month
    result[f'{date_column}_quarter'] = result[date_column].dt.quarter
    result[f'{date_column}_day_of_week'] = result[date_column].dt.dayofweek
    result[f'{date_column}_is_weekend'] = result[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    logger.info(f"Added date features based on column '{date_column}'.")
    return result

def calculate_line_totals(df: pd.DataFrame, quantity_col: str, price_col: str, discount_col: Optional[str] = None) -> pd.DataFrame:
    """Calculate line_total = quantity * price - discount."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if quantity_col not in df.columns:
        raise ValueError(f"Quantity column '{quantity_col}' not found")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
        
    result = df.copy()
    
    # Ensure numeric types
    qty = pd.to_numeric(result[quantity_col], errors='coerce').fillna(0)
    price = pd.to_numeric(result[price_col], errors='coerce').fillna(0)
    base_total = qty * price
    
    if discount_col:
        if discount_col not in result.columns:
            raise ValueError(f"Discount column '{discount_col}' not found")
        discount = pd.to_numeric(result[discount_col], errors='coerce').fillna(0)
        result['line_total'] = base_total - discount
    else:
        result['line_total'] = base_total
        
    logger.info("Calculated line totals.")
    return result

def normalize_column(df: pd.DataFrame, column: str, method: str = 'min_max') -> pd.DataFrame:
    """Normalize a numeric column. Methods: 'min_max', 'z_score'."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
        
    result = df.copy()
    
    if method == 'min_max':
        min_val = result[column].min()
        max_val = result[column].max()
        if max_val == min_val:
            result[f'{column}_normalized'] = 0.0
        else:
            result[f'{column}_normalized'] = (result[column] - min_val) / (max_val - min_val)
    elif method == 'z_score':
        mean_val = result[column].mean()
        std_val = result[column].std()
        if std_val == 0:
            result[f'{column}_normalized'] = 0.0
        else:
            result[f'{column}_normalized'] = (result[column] - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    logger.info(f"Normalized column '{column}' using '{method}' method.")
    return result

def categorize_column(df: pd.DataFrame, column: str, bins: List[float], labels: List[str], new_column: Optional[str] = None) -> pd.DataFrame:
    """Bin a numeric column into categories."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if len(bins) != len(labels) + 1:
        raise ValueError("Number of bins must be exactly one more than number of labels")
        
    result = df.copy()
    output_col = new_column if new_column else f'{column}_category'
    
    result[output_col] = pd.cut(result[column], bins=bins, labels=labels, include_lowest=True)
    logger.info(f"Categorized column '{column}' into '{output_col}'.")
    
    return result
