import os
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def load_revenue_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all required CSVs for revenue analytics.
    Returns dict with keys: orders, order_items, products, regions.
    """
    try:
        orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
        order_items = pd.read_csv(os.path.join(data_dir, "order_items.csv"))
        products = pd.read_csv(os.path.join(data_dir, "products.csv"))
        regions = pd.read_csv(os.path.join(data_dir, "regions.csv"))
        
        # Ensure order_date is datetime
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        
        return {
            "orders": orders,
            "order_items": order_items,
            "products": products,
            "regions": regions
        }
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

def _filter_valid_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Helper to filter out cancelled orders."""
    if orders_df.empty:
        return orders_df
    df = orders_df.copy()
    if 'order_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    return df[df['status'] != 'cancelled'].copy()

def _get_date_filtered_orders(orders_df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Filter orders for the last N days."""
    if orders_df.empty:
        return orders_df
    orders = orders_df.copy()
    if 'order_date' in orders.columns and not pd.api.types.is_datetime64_any_dtype(orders['order_date']):
        orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
    max_date = orders['order_date'].max()
    if pd.isna(max_date):
        return orders
    cutoff = max_date - pd.Timedelta(days=days)
    return orders[orders['order_date'] >= cutoff]

def get_revenue_summary(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    days: int = 30
) -> Dict[str, Any]:
    """Calculate top-level revenue KPIs for the last N days.
    Returns:
    - total_revenue
    - total_orders
    - avg_order_value
    - total_units_sold
    - total_cogs (cost of goods sold = sum of quantity * cost_price)
    - gross_profit (revenue - cogs)
    - gross_margin_pct
    """
    valid_orders = _filter_valid_orders(orders_df)
    period_orders = _get_date_filtered_orders(valid_orders, days)
    
    if period_orders.empty:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "avg_order_value": 0.0,
            "total_units_sold": 0,
            "total_cogs": 0.0,
            "gross_profit": 0.0,
            "gross_margin_pct": 0.0
        }
        
    # Join with items
    items = pd.merge(period_orders[['order_number']], order_items_df, on='order_number', how='inner')
    if items.empty:
        total_revenue = period_orders['total_amount'].sum()
        return {
            "total_revenue": float(total_revenue),
            "total_orders": len(period_orders),
            "avg_order_value": float(total_revenue / len(period_orders)) if len(period_orders) > 0 else 0.0,
            "total_units_sold": 0,
            "total_cogs": 0.0,
            "gross_profit": float(total_revenue),
            "gross_margin_pct": 100.0 if total_revenue > 0 else 0.0
        }
        
    # Add products for COGS
    items = pd.merge(items, products_df[['sku', 'cost_price']], left_on='product_sku', right_on='sku', how='left')
    
    # Fill missing cost price with 0
    items['cost_price'] = items['cost_price'].fillna(0)
    items['discount'] = items['discount'].fillna(0)
    
    items['line_total'] = (items['quantity'] * items['unit_price']) - items['discount']
    items['cogs'] = items['quantity'] * items['cost_price']
    
    total_revenue = items['line_total'].sum()
    total_orders = period_orders['order_number'].nunique()
    total_units_sold = items['quantity'].sum()
    total_cogs = items['cogs'].sum()
    gross_profit = total_revenue - total_cogs
    gross_margin_pct = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0.0
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0.0
    
    return {
        "total_revenue": float(total_revenue),
        "total_orders": int(total_orders),
        "avg_order_value": float(avg_order_value),
        "total_units_sold": int(total_units_sold),
        "total_cogs": float(total_cogs),
        "gross_profit": float(gross_profit),
        "gross_margin_pct": float(gross_margin_pct)
    }

def get_revenue_growth(
    orders_df: pd.DataFrame,
    period_days: int = 30
) -> Dict[str, Any]:
    """Calculate period-over-period revenue growth.
    Compare last period_days vs the period_days before that.
    Returns: current_revenue, previous_revenue, growth_amount, growth_pct
    """
    valid_orders = _filter_valid_orders(orders_df)
    if valid_orders.empty:
         return {
            "current_revenue": 0.0,
            "previous_revenue": 0.0,
            "growth_amount": 0.0,
            "growth_pct": 0.0
        }
    
    max_date = valid_orders['order_date'].max()
    if pd.isna(max_date):
        return {
            "current_revenue": 0.0,
            "previous_revenue": 0.0,
            "growth_amount": 0.0,
            "growth_pct": 0.0
        }
        
    current_cutoff = max_date - pd.Timedelta(days=period_days)
    prev_cutoff = current_cutoff - pd.Timedelta(days=period_days)
    
    current_orders = valid_orders[valid_orders['order_date'] >= current_cutoff]
    prev_orders = valid_orders[(valid_orders['order_date'] >= prev_cutoff) & (valid_orders['order_date'] < current_cutoff)]
    
    current_revenue = current_orders['total_amount'].sum() if not current_orders.empty else 0.0
    previous_revenue = prev_orders['total_amount'].sum() if not prev_orders.empty else 0.0
    
    growth_amount = current_revenue - previous_revenue
    growth_pct = (growth_amount / previous_revenue * 100) if previous_revenue > 0 else (100.0 if current_revenue > 0 else 0.0)
    
    return {
        "current_revenue": float(current_revenue),
        "previous_revenue": float(previous_revenue),
        "growth_amount": float(growth_amount),
        "growth_pct": float(growth_pct)
    }

def get_revenue_by_product(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    days: int = 90,
    top_n: int = 20
) -> pd.DataFrame:
    """Revenue breakdown by product for last N days.
    Returns DataFrame with: product_sku, product_name, category, 
    units_sold, revenue, cogs, profit, margin_pct, rank
    Sorted by revenue descending.
    """
    valid_orders = _filter_valid_orders(orders_df)
    period_orders = _get_date_filtered_orders(valid_orders, days)
    
    empty_df = pd.DataFrame(columns=[
        'product_sku', 'product_name', 'category', 'units_sold', 
        'revenue', 'cogs', 'profit', 'margin_pct', 'rank'
    ])
    
    if period_orders.empty or order_items_df.empty or products_df.empty:
        return empty_df
        
    items = pd.merge(period_orders[['order_number']], order_items_df, on='order_number', how='inner')
    if items.empty:
        return empty_df
        
    cols_to_merge = [c for c in ['sku', 'name', 'category', 'cost_price'] if c in products_df.columns]
    items = pd.merge(items, products_df[cols_to_merge], left_on='product_sku', right_on='sku', how='left')
    
    items['cost_price'] = items['cost_price'].fillna(0)
    items['discount'] = items['discount'].fillna(0)
    
    items['line_total'] = (items['quantity'] * items['unit_price']) - items['discount']
    items['cogs'] = items['quantity'] * items['cost_price']
    
    agg = items.groupby('product_sku').agg(
        product_name=('name', 'first'),
        category=('category', 'first'),
        units_sold=('quantity', 'sum'),
        revenue=('line_total', 'sum'),
        cogs=('cogs', 'sum')
    ).reset_index()
    
    agg['profit'] = agg['revenue'] - agg['cogs']
    agg['margin_pct'] = np.where(agg['revenue'] > 0, (agg['profit'] / agg['revenue']) * 100, 0.0)
    
    agg = agg.sort_values(by='revenue', ascending=False).head(top_n).reset_index(drop=True)
    agg['rank'] = agg.index + 1
    
    return agg

def get_revenue_by_region(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    days: int = 90
) -> pd.DataFrame:
    """Revenue breakdown by region for last N days.
    Returns DataFrame with: region_name, total_orders, revenue, 
    avg_order_value, pct_of_total
    """
    valid_orders = _filter_valid_orders(orders_df)
    period_orders = _get_date_filtered_orders(valid_orders, days)
    
    empty_df = pd.DataFrame(columns=[
        'region_name', 'total_orders', 'revenue', 'avg_order_value', 'pct_of_total'
    ])
    
    if period_orders.empty:
        return empty_df
        
    items = pd.merge(period_orders, order_items_df, on='order_number', how='left')
    items['discount'] = items['discount'].fillna(0)
    
    if items.empty or order_items_df.empty:
        agg = period_orders.groupby('region_name').agg(
            total_orders=('order_number', 'nunique'),
            revenue=('total_amount', 'sum')
        ).reset_index()
    else:
        items['line_total'] = (items['quantity'] * items['unit_price']) - items['discount']
        order_totals = items.groupby(['order_number', 'region_name']).agg(
            order_revenue=('line_total', 'sum')
        ).reset_index()
        
        agg = order_totals.groupby('region_name').agg(
            total_orders=('order_number', 'nunique'),
            revenue=('order_revenue', 'sum')
        ).reset_index()
        
    agg['avg_order_value'] = np.where(agg['total_orders'] > 0, agg['revenue'] / agg['total_orders'], 0.0)
    total_rev = agg['revenue'].sum()
    agg['pct_of_total'] = np.where(total_rev > 0, (agg['revenue'] / total_rev) * 100, 0.0)
    
    return agg.sort_values(by='revenue', ascending=False).reset_index(drop=True)

def get_revenue_by_customer(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """Top customers by revenue.
    Returns: customer_email, total_orders, total_revenue, avg_order_value, 
    first_order, last_order
    """
    valid_orders = _filter_valid_orders(orders_df)
    
    empty_df = pd.DataFrame(columns=[
        'customer_email', 'total_orders', 'total_revenue', 'avg_order_value', 'first_order', 'last_order'
    ])
    
    if valid_orders.empty:
         return empty_df
         
    items = pd.merge(valid_orders, order_items_df, on='order_number', how='left')
    items['discount'] = items['discount'].fillna(0)
    
    if items.empty or order_items_df.empty:
        agg = valid_orders.groupby('customer_email').agg(
            total_orders=('order_number', 'nunique'),
            total_revenue=('total_amount', 'sum'),
            first_order=('order_date', 'min'),
            last_order=('order_date', 'max')
        ).reset_index()
    else:
        items['line_total'] = (items['quantity'] * items['unit_price']) - items['discount']
        order_totals = items.groupby(['order_number', 'customer_email', 'order_date']).agg(
            order_revenue=('line_total', 'sum')
        ).reset_index()
        
        agg = order_totals.groupby('customer_email').agg(
            total_orders=('order_number', 'nunique'),
            total_revenue=('order_revenue', 'sum'),
            first_order=('order_date', 'min'),
            last_order=('order_date', 'max')
        ).reset_index()
    
    agg['avg_order_value'] = np.where(agg['total_orders'] > 0, agg['total_revenue'] / agg['total_orders'], 0.0)
    
    return agg.sort_values(by='total_revenue', ascending=False).head(top_n).reset_index(drop=True)

def get_product_performance(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> pd.DataFrame:
    """Comprehensive product performance metrics.
    Returns: sku, name, category, units_sold, revenue, cogs, profit, 
    margin_pct, avg_selling_price, order_count
    """
    valid_orders = _filter_valid_orders(orders_df)
    
    empty_df = pd.DataFrame(columns=[
        'sku', 'name', 'category', 'units_sold', 'revenue', 
        'cogs', 'profit', 'margin_pct', 'avg_selling_price', 'order_count'
    ])
    
    if valid_orders.empty or order_items_df.empty or products_df.empty:
        return empty_df
        
    items = pd.merge(valid_orders[['order_number']], order_items_df, on='order_number', how='inner')
    if items.empty:
        return empty_df
        
    cols_to_merge = [c for c in ['sku', 'name', 'category', 'cost_price'] if c in products_df.columns]
    items = pd.merge(items, products_df[cols_to_merge], left_on='product_sku', right_on='sku', how='left')
    
    items['cost_price'] = items['cost_price'].fillna(0)
    items['discount'] = items['discount'].fillna(0)
    
    items['line_total'] = (items['quantity'] * items['unit_price']) - items['discount']
    items['cogs'] = items['quantity'] * items['cost_price']
    
    agg = items.groupby('product_sku').agg(
        name=('name', 'first'),
        category=('category', 'first'),
        units_sold=('quantity', 'sum'),
        revenue=('line_total', 'sum'),
        cogs=('cogs', 'sum'),
        order_count=('order_number', 'nunique')
    ).reset_index().rename(columns={'product_sku': 'sku'})
    
    agg['profit'] = agg['revenue'] - agg['cogs']
    agg['margin_pct'] = np.where(agg['revenue'] > 0, (agg['profit'] / agg['revenue']) * 100, 0.0)
    agg['avg_selling_price'] = np.where(agg['units_sold'] > 0, agg['revenue'] / agg['units_sold'], 0.0)
    
    return agg.sort_values(by='revenue', ascending=False).reset_index(drop=True)

def get_monthly_revenue_trend(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    months: int = 12
) -> pd.DataFrame:
    """Monthly revenue trend for the last N months.
    Returns: month, revenue, orders, units_sold, avg_order_value
    """
    valid_orders = _filter_valid_orders(orders_df)
    
    empty_df = pd.DataFrame(columns=['month', 'revenue', 'orders', 'units_sold', 'avg_order_value'])
    
    if valid_orders.empty:
        return empty_df
        
    max_date = valid_orders['order_date'].max()
    if pd.isna(max_date):
        return empty_df
        
    cutoff = max_date - pd.DateOffset(months=months)
    period_orders = valid_orders[valid_orders['order_date'] >= cutoff].copy()
    
    if period_orders.empty:
        return empty_df
        
    period_orders['month'] = period_orders['order_date'].dt.to_period('M').astype(str)
    
    items = pd.merge(period_orders, order_items_df, on='order_number', how='left')
    items['discount'] = items['discount'].fillna(0)
    
    if items.empty or order_items_df.empty:
        agg = period_orders.groupby('month').agg(
            revenue=('total_amount', 'sum'),
            orders=('order_number', 'nunique')
        ).reset_index()
        agg['units_sold'] = 0
    else:
        items['line_total'] = (items['quantity'] * items['unit_price']) - items['discount']
        agg = items.groupby('month').agg(
            revenue=('line_total', 'sum'),
            orders=('order_number', 'nunique'),
            units_sold=('quantity', 'sum')
        ).reset_index()
    
    # Sort chronologically
    agg = agg.sort_values(by='month')
    agg['avg_order_value'] = np.where(agg['orders'] > 0, agg['revenue'] / agg['orders'], 0.0)
    
    return agg.reset_index(drop=True)
