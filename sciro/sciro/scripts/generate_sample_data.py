import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure output directory
OUTPUT_DIR = os.environ.get('SCIRO_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'sample'))

# Set random seed for reproducibility
rng = np.random.default_rng(42)

def generate_regions():
    logger.info("Generating regions...")
    regions_data = [
        {'name': 'North', 'country': 'India'},
        {'name': 'South', 'country': 'India'},
        {'name': 'East', 'country': 'India'},
        {'name': 'West', 'country': 'India'},
        {'name': 'Central', 'country': 'India'},
        {'name': 'North-East', 'country': 'India'},
        {'name': 'North-West', 'country': 'India'},
        {'name': 'South-East', 'country': 'India'},
        {'name': 'South-West', 'country': 'India'},
        {'name': 'Pan-India', 'country': 'India'}
    ]
    return pd.DataFrame(regions_data)

def generate_suppliers(regions_df):
    logger.info("Generating suppliers...")
    prefixes = ['Tata', 'Reliance', 'Birla', 'Adani', 'JSW', 'Mahindra', 'Godrej', 'Larsen', 'Essar', 'Vedanta', 'Jindal', 'Hindalco', 'Bharat']
    suffixes = ['Steel', 'Materials', 'Supplies', 'Industries', 'Corporation', 'Logistics', 'Trading', 'Enterprises', 'Global', 'Solutions']
    
    suppliers = []
    for i in range(50):
        # Generate realistic Indian company names
        name = f"{rng.choice(prefixes)} {rng.choice(suffixes)} {rng.choice(['Ltd', 'Pvt Ltd', 'Inc', 'Co'])} {i}"
        
        email_domain = name.lower().replace(' ', '').replace('.', '') + '.com'
        email = f"contact@{email_domain}"
        
        # Indian phone number format
        phone = f"+91 {rng.choice(['9', '8', '7'])}{''.join([str(rng.integers(0, 10)) for _ in range(9)])}"
        
        region = rng.choice(regions_df['name'])
        
        # Generate some low reliability suppliers for risk analysis (~20%)
        is_risky = rng.random() < 0.20
        if is_risky:
            reliability = rng.uniform(0.5, 0.69)
            lead_time = rng.integers(15, 30)
        else:
            reliability = rng.uniform(0.75, 1.0)
            lead_time = rng.integers(3, 15)
            
        status = rng.choice(['active', 'inactive'], p=[0.9, 0.1])
        
        suppliers.append({
            'name': name,
            'contact_email': email,
            'contact_phone': phone,
            'region_name': region,
            'lead_time_days': lead_time,
            'reliability_score': round(reliability, 2),
            'status': status
        })
        
    return pd.DataFrame(suppliers)

def generate_products(suppliers_df):
    logger.info("Generating products...")
    categories = ['Electronics', 'Raw Materials', 'Packaging', 'Chemicals', 'Textiles', 'Machinery', 'Components', 'Consumables']
    cat_prefixes = {'Electronics': 'ELEC', 'Raw Materials': 'RMAT', 'Packaging': 'PACK', 'Chemicals': 'CHEM', 
                    'Textiles': 'TEXT', 'Machinery': 'MACH', 'Components': 'COMP', 'Consumables': 'CONS'}
    
    products = []
    cat_counts = {cat: 1 for cat in categories}
    
    # Use only active suppliers
    active_suppliers = suppliers_df[suppliers_df['status'] == 'active']['name'].tolist()
    if not active_suppliers:
        active_suppliers = suppliers_df['name'].tolist() # Fallback
    
    for _ in range(200):
        category = rng.choice(categories)
        prefix = cat_prefixes[category]
        sku = f"{prefix}-{cat_counts[category]:04d}"
        cat_counts[category] += 1
        
        name = f"{category} Product {cat_counts[category]}"
        unit_price = rng.uniform(100, 50000)
        # Ensure cost_price < unit_price always (60-80%)
        cost_price = unit_price * rng.uniform(0.6, 0.8)
        
        weight = rng.uniform(0.1, 500)
        supplier = rng.choice(active_suppliers)
        reorder_point = rng.integers(20, 200)
        safety_stock = rng.integers(10, 100)
        status = rng.choice(['active', 'discontinued'], p=[0.95, 0.05])
        
        products.append({
            'sku': sku,
            'name': name,
            'category': category,
            'unit_price': round(unit_price, 2),
            'cost_price': round(cost_price, 2),
            'weight_kg': round(weight, 2),
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'supplier_name': supplier,
            'status': status
        })
        
    return pd.DataFrame(products)

def generate_inventory(products_df, regions_df):
    logger.info("Generating inventory...")
    inventory = []
    
    for sku in products_df['sku']:
        # Not all products in all regions (pick a subset of regions)
        num_regions = rng.integers(2, len(regions_df) + 1)
        prod_regions = rng.choice(regions_df['name'], size=num_regions, replace=False)
        
        for region in prod_regions:
            stock_scenario = rng.random()
            if stock_scenario < 0.10: # ~10% very low stock <20
                qty_on_hand = rng.integers(1, 20)
            elif stock_scenario < 0.15: # ~5% out of stock
                qty_on_hand = 0
            else:
                qty_on_hand = rng.integers(20, 2000)
                
            qty_reserved = int(qty_on_hand * rng.uniform(0, 0.3)) if qty_on_hand > 0 else 0
            
            inventory.append({
                'product_sku': sku,
                'region_name': region,
                'quantity_on_hand': qty_on_hand,
                'quantity_reserved': qty_reserved
            })
            
    return pd.DataFrame(inventory)

def generate_shipments(suppliers_df, products_df, regions_df):
    logger.info("Generating shipments...")
    shipments = []
    carriers = ['BlueDart', 'DTDC', 'Delhivery', 'Gati', 'XpressBees']
    statuses = ['pending', 'in_transit', 'delivered', 'delayed', 'cancelled']
    status_probs = [0.05, 0.10, 0.70, 0.10, 0.05]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    supplier_dict = suppliers_df.set_index('name').to_dict('index')
    product_skus = products_df['sku'].tolist()
    
    for i in range(1000):
        supplier_name = rng.choice(suppliers_df['name'])
        supplier_info = supplier_dict[supplier_name]
        
        # Random ship date in the last year
        shipped_date = start_date + timedelta(days=float(rng.uniform(0, 365)))
        expected_delivery = shipped_date + timedelta(days=int(supplier_info['lead_time_days'])) + timedelta(days=int(rng.integers(0, 6)))
        
        status = rng.choice(statuses, p=status_probs)
        
        actual_delivery = pd.NaT
        if status == 'delivered':
            # Late delivery correlates with low reliability
            is_late = rng.random() > supplier_info['reliability_score']
            if is_late:
                actual_delivery = expected_delivery + timedelta(days=int(rng.integers(1, 16)))
            else:
                actual_delivery = expected_delivery - timedelta(days=int(rng.integers(0, 3)))
        elif status == 'delayed':
            actual_delivery = pd.NaT
            
        tracking = f"TRK-{shipped_date.strftime('%Y%m%d')}-{i:04d}"
        
        shipments.append({
            'tracking_number': tracking,
            'supplier_name': supplier_name,
            'product_sku': rng.choice(product_skus),
            'destination_region': rng.choice(regions_df['name']),
            'quantity': int(rng.integers(10, 500)),
            'status': status,
            'carrier': rng.choice(carriers),
            'shipping_cost': round(rng.uniform(200, 5000), 2),
            'shipped_date': shipped_date.strftime('%Y-%m-%d'),
            'expected_delivery_date': expected_delivery.strftime('%Y-%m-%d'),
            'actual_delivery_date': actual_delivery.strftime('%Y-%m-%d') if pd.notna(actual_delivery) else ''
        })
        
    return pd.DataFrame(shipments)

def generate_customers(regions_df):
    logger.info("Generating customers...")
    first_names = ['Amit', 'Rahul', 'Priya', 'Neha', 'Sanjay', 'Vikram', 'Anjali', 'Kiran', 'Raj', 'Sunil', 'Pooja', 'Deepak', 'Arjun', 'Kavita']
    last_names = ['Sharma', 'Patel', 'Singh', 'Kumar', 'Gupta', 'Verma', 'Reddy', 'Jain', 'Mehta', 'Nair', 'Deshmukh', 'Iyer']
    company_types = ['Tech', 'Solutions', 'Enterprises', 'Retail', 'Consulting', 'Manufacturing', 'Services']
    
    customers = []
    for i in range(300):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        name = f"{first} {last} {i}"
        
        company = f"{first} {rng.choice(company_types)}"
        email = f"{first.lower()}.{last.lower()}{i}@{company.lower().replace(' ', '')}.com"
        
        tier = rng.choice(['standard', 'gold', 'enterprise'], p=[0.6, 0.3, 0.1])
        
        customers.append({
            'name': name,
            'email': email,
            'company_name': company,
            'region_name': rng.choice(regions_df['name']),
            'tier': tier
        })
        
    return pd.DataFrame(customers)

def generate_orders_and_items(customers_df, products_df):
    logger.info("Generating orders and items...")
    orders = []
    order_items = []
    
    statuses = ['pending', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled']
    status_probs = [0.05, 0.10, 0.05, 0.15, 0.60, 0.05]
    
    end_date = datetime.now()
    
    product_dict = products_df.set_index('sku').to_dict('index')
    product_skus = products_df['sku'].tolist()
    
    for i in range(5000):
        # Seasonality: more orders in Q3-Q4
        month_probs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1]
        month = rng.choice(list(range(1, 13)), p=month_probs)
        year = end_date.year if month <= end_date.month else end_date.year - 1
        day = rng.integers(1, 28)
        order_date = datetime(year, month, day)
        
        customer = customers_df.iloc[rng.integers(0, len(customers_df))]
        order_number = f"ORD-{order_date.strftime('%Y%m')}-{i:05d}"
        status = rng.choice(statuses, p=status_probs)
        
        if status == 'cancelled':
            pay_status = rng.choice(['unpaid', 'refunded'])
        elif status in ['shipped', 'delivered']:
            pay_status = 'paid'
        else:
            pay_status = rng.choice(['unpaid', 'paid'], p=[0.4, 0.6])
            
        num_items = rng.integers(1, 6)
        order_total = 0
        
        selected_skus = rng.choice(product_skus, size=num_items, replace=False)
        for sku in selected_skus:
            prod_info = product_dict[sku]
            qty = rng.integers(1, 51)
            
            # Unit price from product with slight variation
            variation = rng.uniform(0.95, 1.05)
            unit_price = round(prod_info['unit_price'] * variation, 2)
            
            # Discount as absolute value (0-10% of gross)
            gross = unit_price * qty
            discount_amt = round(gross * rng.uniform(0, 0.1), 2)
            item_total = round(gross - discount_amt, 2)
            order_total += item_total
            
            order_items.append({
                'order_number': order_number,
                'product_sku': sku,
                'quantity': qty,
                'unit_price': unit_price,
                'discount': discount_amt
            })
            
        orders.append({
            'order_number': order_number,
            'customer_email': customer['email'],
            'region_name': customer['region_name'],
            'order_date': order_date.strftime('%Y-%m-%d'),
            'status': status,
            'payment_status': pay_status,
            'total_amount': round(order_total, 2)
        })
        
    return pd.DataFrame(orders), pd.DataFrame(order_items)

def main():
    logger.info(f"Starting data generation. Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    regions_df = generate_regions()
    regions_df.to_csv(os.path.join(OUTPUT_DIR, 'regions.csv'), index=False)
    
    suppliers_df = generate_suppliers(regions_df)
    suppliers_df.to_csv(os.path.join(OUTPUT_DIR, 'suppliers.csv'), index=False)
    
    products_df = generate_products(suppliers_df)
    products_df.to_csv(os.path.join(OUTPUT_DIR, 'products.csv'), index=False)
    
    inventory_df = generate_inventory(products_df, regions_df)
    inventory_df.to_csv(os.path.join(OUTPUT_DIR, 'inventory.csv'), index=False)
    
    shipments_df = generate_shipments(suppliers_df, products_df, regions_df)
    shipments_df.to_csv(os.path.join(OUTPUT_DIR, 'shipments.csv'), index=False)
    
    customers_df = generate_customers(regions_df)
    customers_df.to_csv(os.path.join(OUTPUT_DIR, 'customers.csv'), index=False)
    
    orders_df, order_items_df = generate_orders_and_items(customers_df, products_df)
    orders_df.to_csv(os.path.join(OUTPUT_DIR, 'orders.csv'), index=False)
    order_items_df.to_csv(os.path.join(OUTPUT_DIR, 'order_items.csv'), index=False)
    
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Regions generated:     {len(regions_df)}")
    print(f"Suppliers generated:   {len(suppliers_df)}")
    print(f"Products generated:    {len(products_df)}")
    print(f"Inventory records:     {len(inventory_df)}")
    print(f"Shipments generated:   {len(shipments_df)}")
    print(f"Customers generated:   {len(customers_df)}")
    print(f"Orders generated:      {len(orders_df)}")
    print(f"Order items generated: {len(order_items_df)}")
    print("="*50)
    print(f"All files saved to: {OUTPUT_DIR}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
