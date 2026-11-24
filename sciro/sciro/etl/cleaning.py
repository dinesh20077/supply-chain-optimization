import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

def handle_missing_values(
    df: pd.DataFrame,
    strategies: Dict[str, str],
    fill_values: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Handle missing values with configurable strategies per column.
    
    strategies dict maps column_name -> strategy:
    - 'drop': drop rows with missing values in this column
    - 'mean': fill with column mean (numeric only)
    - 'median': fill with column median (numeric only)
    - 'mode': fill with most frequent value
    - 'ffill': forward fill
    - 'bfill': backward fill
    - 'default': fill with value from fill_values dict
    
    Returns cleaned DataFrame and logs what was done.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    result = df.copy()
    fill_values = fill_values or {}
    
    for col, strategy in strategies.items():
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
            logger.info(f"Dropped rows with missing values in column '{col}'.")
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
                logger.info(f"Filled missing values in '{col}' with mean.")
            else:
                logger.error(f"Cannot apply 'mean' strategy to non-numeric column '{col}'.")
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
                logger.info(f"Filled missing values in '{col}' with median.")
            else:
                logger.error(f"Cannot apply 'median' strategy to non-numeric column '{col}'.")
        elif strategy == 'mode':
            if not result[col].mode().empty:
                result[col] = result[col].fillna(result[col].mode()[0])
                logger.info(f"Filled missing values in '{col}' with mode.")
        elif strategy == 'ffill':
            result[col] = result[col].ffill()
            logger.info(f"Forward filled missing values in '{col}'.")
        elif strategy == 'bfill':
            result[col] = result[col].bfill()
            logger.info(f"Backward filled missing values in '{col}'.")
        elif strategy == 'default':
            if col in fill_values:
                result[col] = result[col].fillna(fill_values[col])
                logger.info(f"Filled missing values in '{col}' with default value.")
            else:
                logger.error(f"No fill value provided for column '{col}' with 'default' strategy.")
        else:
            logger.warning(f"Unknown strategy '{strategy}' for column '{col}'. Skipping.")
            
    return result

def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and separate duplicates.
    Returns (clean_df, duplicates_df)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    mask = df.duplicated(subset=subset, keep=keep)
    duplicates_df = df[mask].copy()
    clean_df = df[~mask].copy()
    
    logger.info(f"Detected {len(duplicates_df)} duplicates.")
    return clean_df, duplicates_df

def standardize_text_columns(
    df: pd.DataFrame,
    columns: List[str],
    operations: List[str] = ['strip', 'lower']
) -> pd.DataFrame:
    """Standardize text columns: strip whitespace, normalize case.
    operations can include: 'strip', 'lower', 'upper', 'title'
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found. Skipping.")
            continue
            
        if not pd.api.types.is_string_dtype(result[col]) and not pd.api.types.is_object_dtype(result[col]):
            logger.warning(f"Column '{col}' is not a string type. Skipping text standardization.")
            continue
            
        for op in operations:
            if op == 'strip':
                result[col] = result[col].astype(str).str.strip()
            elif op == 'lower':
                result[col] = result[col].astype(str).str.lower()
            elif op == 'upper':
                result[col] = result[col].astype(str).str.upper()
            elif op == 'title':
                result[col] = result[col].astype(str).str.title()
            else:
                logger.warning(f"Unknown operation '{op}' for text standardization.")
                
        logger.info(f"Standardized text column '{col}' with operations: {operations}.")
                
    return result

def coerce_types(
    df: pd.DataFrame,
    type_map: Dict[str, str]
) -> pd.DataFrame:
    """Coerce column types safely.
    type_map maps column_name -> target_type ('int', 'float', 'datetime', 'str', 'bool')
    Logs warnings for failed conversions.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    result = df.copy()
    for col, target_type in type_map.items():
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found for type coercion. Skipping.")
            continue
            
        try:
            if target_type == 'datetime':
                result[col] = pd.to_datetime(result[col])
            else:
                result[col] = result[col].astype(target_type)
            logger.info(f"Coerced column '{col}' to type '{target_type}'.")
        except Exception as e:
            logger.warning(f"Failed to convert column '{col}' to type '{target_type}': {e}")
            
    return result

def clean_dataframe(
    df: pd.DataFrame,
    duplicate_subset: Optional[List[str]] = None,
    missing_strategies: Optional[Dict[str, str]] = None,
    fill_values: Optional[Dict[str, Any]] = None,
    text_columns: Optional[List[str]] = None,
    type_map: Optional[Dict[str, str]] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Full cleaning pipeline. Returns (cleaned_df, cleaning_report).
    cleaning_report contains: rows_before, rows_after, duplicates_removed, 
    missing_values_handled, type_conversions
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
        
    logger.info("Starting full DataFrame cleaning pipeline.")
    original_rows = len(df)
    result = df.copy()
    report: Dict[str, Any] = {
        'rows_before': original_rows,
        'duplicates_removed': 0,
        'missing_values_handled': {},
        'type_conversions': type_map or {}
    }
    
    # 1. Detect and remove duplicates
    result, duplicates = detect_duplicates(result, subset=duplicate_subset)
    report['duplicates_removed'] = len(duplicates)
    
    # 2. Handle missing values
    if missing_strategies:
        missing_before = result.isna().sum().to_dict()
        result = handle_missing_values(result, missing_strategies, fill_values)
        missing_after = result.isna().sum().to_dict()
        for col in missing_strategies:
            if col in missing_before and col in missing_after:
                handled = missing_before[col] - missing_after[col]
                report['missing_values_handled'][col] = handled
                
    # 3. Standardize text
    if text_columns:
        result = standardize_text_columns(result, text_columns)
        
    # 4. Coerce types
    if type_map:
        result = coerce_types(result, type_map)
        
    report['rows_after'] = len(result)
    logger.info(f"Cleaning pipeline completed. Reduced rows from {original_rows} to {len(result)}.")
    
    return result, report
