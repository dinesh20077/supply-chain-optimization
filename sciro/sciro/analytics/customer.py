import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_customer_metrics(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate per-customer metrics.
    Returns DataFrame with: email, name, company_name, region_name, tier,
    total_orders, delivered_orders, cancelled_orders, 
    total_revenue, avg_order_value, max_order_value,
    first_order_date, last_order_date, days_since_last_order,
    order_frequency_days (avg days between orders)
    Filter out cancelled orders for revenue calculations.
    """
    if customers_df.empty:
        return pd.DataFrame()

    orders = orders_df.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
    
    items = order_items_df.copy()
    items['item_revenue'] = (items['quantity'] * items['unit_price']) - items['discount'].fillna(0)
    order_revenue = items.groupby('order_number')['item_revenue'].sum().reset_index()
    order_revenue.rename(columns={'item_revenue': 'calculated_total'}, inplace=True)
    
    orders = orders.merge(order_revenue, on='order_number', how='left')
    orders['calculated_total'] = orders['calculated_total'].fillna(0)
    
    active_orders = orders[orders['status'].str.lower() != 'cancelled']
    current_date = orders['order_date'].max() if not orders.empty else pd.Timestamp.now()
    
    metrics = []
    for _, customer in customers_df.iterrows():
        email = customer['email']
        cust_orders = orders[orders['customer_email'] == email]
        cust_active = active_orders[active_orders['customer_email'] == email]
        
        total_orders = len(cust_orders)
        delivered_orders = len(cust_orders[cust_orders['status'].str.lower() == 'delivered'])
        cancelled_orders = len(cust_orders[cust_orders['status'].str.lower() == 'cancelled'])
        
        total_revenue = cust_active['calculated_total'].sum()
        avg_order_value = cust_active['calculated_total'].mean() if not cust_active.empty else 0.0
        max_order_value = cust_active['calculated_total'].max() if not cust_active.empty else 0.0
        
        first_order_date = cust_orders['order_date'].min() if not cust_orders.empty else pd.NaT
        last_order_date = cust_orders['order_date'].max() if not cust_orders.empty else pd.NaT
        
        days_since_last_order = (current_date - last_order_date).days if pd.notnull(last_order_date) else None
        
        order_freq = None
        if total_orders > 1 and pd.notnull(first_order_date) and pd.notnull(last_order_date):
            order_freq = (last_order_date - first_order_date).days / (total_orders - 1)
            
        metrics.append({
            'email': email,
            'name': customer.get('name'),
            'company_name': customer.get('company_name'),
            'region_name': customer.get('region_name'),
            'tier': customer.get('tier'),
            'total_orders': total_orders,
            'delivered_orders': delivered_orders,
            'cancelled_orders': cancelled_orders,
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'max_order_value': max_order_value,
            'first_order_date': first_order_date,
            'last_order_date': last_order_date,
            'days_since_last_order': days_since_last_order,
            'order_frequency_days': order_freq
        })
        
    return pd.DataFrame(metrics)

def calculate_customer_ltv(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    lifespan_years: float = 2.0
) -> pd.DataFrame:
    """Calculate Customer Lifetime Value.
    CLV = avg_order_value * purchase_frequency_per_year * lifespan_years
    purchase_frequency_per_year = total_orders / customer_tenure_years (min 0.5 yr)
    Returns DataFrame with: email, name, tier, total_revenue, avg_order_value,
    orders_per_year, estimated_clv, clv_tier (High/Medium/Low)
    """
    metrics_df = get_customer_metrics(customers_df, orders_df, order_items_df)
    if metrics_df.empty:
        return pd.DataFrame()
        
    current_date = pd.to_datetime(orders_df['order_date']).max() if not orders_df.empty else pd.Timestamp.now()
    
    clv_data = []
    for _, row in metrics_df.iterrows():
        if pd.isnull(row['first_order_date']):
            tenure_years = 0.5
        else:
            tenure_days = (current_date - row['first_order_date']).days
            tenure_years = max(0.5, tenure_days / 365.25)
            
        orders_per_year = row['total_orders'] / tenure_years
        estimated_clv = row['avg_order_value'] * orders_per_year * lifespan_years
        
        clv_data.append({
            'email': row['email'],
            'name': row['name'],
            'tier': row['tier'],
            'total_revenue': row['total_revenue'],
            'avg_order_value': row['avg_order_value'],
            'orders_per_year': orders_per_year,
            'estimated_clv': estimated_clv
        })
        
    df = pd.DataFrame(clv_data)
    
    if not df.empty:
        # Determine clv_tier
        p33 = df['estimated_clv'].quantile(0.33)
        p67 = df['estimated_clv'].quantile(0.67)
        
        def assign_tier(val):
            if val >= p67: return 'High'
            elif val >= p33: return 'Medium'
            else: return 'Low'
            
        df['clv_tier'] = df['estimated_clv'].apply(assign_tier)
    else:
        df['clv_tier'] = pd.Series(dtype='object')
        
    return df

def segment_customers(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame
) -> pd.DataFrame:
    """RFM-based customer segmentation.
    
    R (Recency): days since last order. Lower = better. Score 1-5.
    F (Frequency): total orders. Higher = better. Score 1-5.
    M (Monetary): total revenue. Higher = better. Score 1-5.
    
    Use pd.qcut for scoring (5 bins).
    
    Segment mapping based on RFM scores:
    - Champions: R>=4, F>=4, M>=4
    - Loyal Customers: F>=3, M>=3
    - Potential Loyalists: R>=4, F>=2
    - At Risk: R<=2, F>=3
    - Lost: R<=2, F<=2
    - Others: everything else labeled 'Regular'
    
    Returns DataFrame with: email, name, tier, recency_days, frequency, 
    monetary, r_score, f_score, m_score, rfm_score, segment
    """
    metrics = get_customer_metrics(customers_df, orders_df, order_items_df)
    if metrics.empty:
        return pd.DataFrame()
        
    # Fill NAs for missing orders
    max_recency = metrics['days_since_last_order'].max()
    max_recency = max_recency if pd.notnull(max_recency) else 9999
    
    df = metrics[['email', 'name', 'tier', 'days_since_last_order', 'total_orders', 'total_revenue']].copy()
    df.rename(columns={
        'days_since_last_order': 'recency_days',
        'total_orders': 'frequency',
        'total_revenue': 'monetary'
    }, inplace=True)
    df['recency_days'] = df['recency_days'].fillna(max_recency + 100)
    
    def calculate_score(series, labels, duplicates_drop=True):
        try:
            return pd.qcut(series, q=5, labels=labels, duplicates='drop')
        except ValueError:
            return pd.cut(series, bins=5, labels=labels)
            
    # R: lower is better, so labels are 5 to 1
    df['r_score'] = calculate_score(df['recency_days'], labels=[5, 4, 3, 2, 1]).astype(int)
    # F: higher is better, labels 1 to 5
    df['f_score'] = calculate_score(df['frequency'], labels=[1, 2, 3, 4, 5]).astype(int)
    # M: higher is better, labels 1 to 5
    df['m_score'] = calculate_score(df['monetary'], labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['rfm_score'] = df['r_score'].astype(str) + df['f_score'].astype(str) + df['m_score'].astype(str)
    
    def assign_segment(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f >= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Regular'
            
    df['segment'] = df.apply(assign_segment, axis=1)
    return df

def calculate_churn_probability(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame
) -> pd.DataFrame:
    """Estimate churn probability based on recency and frequency.
    
    Logic:
    - Calculate avg purchase interval per customer
    - If days_since_last_order > 2 * avg_interval -> high churn risk
    - If days_since_last_order > 1.5 * avg_interval -> medium
    - Otherwise low
    
    churn_score = min(1.0, days_since_last / (2 * avg_interval))
    For customers with only 1 order, use 90 days as default interval.
    
    Returns: email, name, days_since_last_order, avg_purchase_interval,
    churn_score (0-1), churn_risk (Low/Medium/High)
    """
    if customers_df.empty:
        return pd.DataFrame()
        
    orders = orders_df.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
    current_date = orders['order_date'].dropna().max() if not orders['order_date'].dropna().empty else pd.Timestamp.now()
    
    results = []
    for _, cust in customers_df.iterrows():
        email = cust['email']
        cust_orders = orders[orders['customer_email'] == email].sort_values('order_date')
        
        last_order_date = cust_orders['order_date'].max() if not cust_orders.empty else pd.NaT
        days_since_last = (current_date - last_order_date).days if pd.notnull(last_order_date) else None
        
        avg_interval = 90.0
        if len(cust_orders) > 1:
            first_order = cust_orders['order_date'].min()
            avg_interval = (last_order_date - first_order).days / (len(cust_orders) - 1)
            if avg_interval <= 0:
                avg_interval = 90.0
                
        if days_since_last is None:
            churn_score = 0.0
            churn_risk = 'Low'
            days_since_last = 0
        else:
            churn_score = min(1.0, days_since_last / (2 * avg_interval))
            if days_since_last > 2 * avg_interval:
                churn_risk = 'High'
            elif days_since_last > 1.5 * avg_interval:
                churn_risk = 'Medium'
            else:
                churn_risk = 'Low'
                
        results.append({
            'email': email,
            'name': cust.get('name'),
            'days_since_last_order': days_since_last,
            'avg_purchase_interval': avg_interval,
            'churn_score': churn_score,
            'churn_risk': churn_risk
        })
        
    return pd.DataFrame(results)

def get_customer_summary(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame
) -> Dict[str, Any]:
    """High-level customer KPIs.
    Returns dict:
    - total_customers
    - active_customers (ordered in last 90 days)
    - new_customers (first order in last 30 days)
    - avg_customer_ltv
    - customers_by_tier (dict)
    - customers_by_segment (dict)
    - top_customers (list of top 5 by revenue)
    - at_risk_customers (count with high churn probability)
    """
    summary: Dict[str, Any] = {
        'total_customers': 0,
        'active_customers': 0,
        'new_customers': 0,
        'avg_customer_ltv': 0.0,
        'customers_by_tier': {},
        'customers_by_segment': {},
        'top_customers': [],
        'at_risk_customers': 0
    }
    
    if customers_df.empty:
        return summary
        
    summary['total_customers'] = len(customers_df)
    summary['customers_by_tier'] = customers_df['tier'].value_counts().to_dict() if 'tier' in customers_df.columns else {}
    
    metrics = get_customer_metrics(customers_df, orders_df, order_items_df)
    if not metrics.empty:
        summary['active_customers'] = len(metrics[metrics['days_since_last_order'] <= 90])
        
        parsed_dates = pd.to_datetime(orders_df['order_date'], errors='coerce').dropna()
        current_date = parsed_dates.max() if not parsed_dates.empty else pd.Timestamp.now()
        new_cust_thresh = current_date - pd.Timedelta(days=30)
        summary['new_customers'] = len(metrics[metrics['first_order_date'] >= new_cust_thresh])
        
        top_5 = metrics.nlargest(5, 'total_revenue')
        summary['top_customers'] = top_5[['email', 'name', 'total_revenue']].to_dict('records')
        
    ltv = calculate_customer_ltv(customers_df, orders_df, order_items_df)
    if not ltv.empty:
        summary['avg_customer_ltv'] = ltv['estimated_clv'].mean()
        
    segments = segment_customers(customers_df, orders_df, order_items_df)
    if not segments.empty:
        summary['customers_by_segment'] = segments['segment'].value_counts().to_dict()
        
    churn = calculate_churn_probability(customers_df, orders_df)
    if not churn.empty:
        summary['at_risk_customers'] = len(churn[churn['churn_risk'] == 'High'])
        
    return summary
