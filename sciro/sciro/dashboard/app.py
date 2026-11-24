"""
SCIRO Streamlit Dashboard — Main Application.

Supply Chain Intelligence & Revenue Optimization Platform.
Run with: streamlit run sciro/dashboard/app.py

Pages:
1. Executive Dashboard — KPI overview
2. Supply Chain — Supplier performance, risk scores, shipment analysis
3. Inventory — Stock health, stockout risk, turnover
4. Revenue — Revenue KPIs, trends, product performance
5. Customers — Segmentation, LTV, churn risk
6. ML Models — Train/predict with demand, revenue, stockout, supplier risk models
7. Simulation — Monte Carlo disruption, pricing, inventory optimization
8. AI Copilot — Natural language query assistant
"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sciro.config.settings import get_settings
from sciro.dashboard.auth import AuthManager
from sciro.ingestion.loader import (
    load_file,
    load_excel,
    load_pdf,
    load_csv,
    get_excel_sheet_names,
    detect_format,
)
from sciro.ingestion.validators import validate_dataframe
from sciro.etl.cleaning import clean_dataframe

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="SCIRO — Supply Chain Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    /* Main theme */
    .main .block-container { padding-top: 1rem; }
    .stMetric { background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                padding: 1rem; border-radius: 12px; border: 1px solid #475569; }
    .stMetric label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.8rem !important; }
    .stMetric [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }

    /* Cards */
    .card { background: #1e293b; border-radius: 12px; padding: 1.5rem;
            border: 1px solid #334155; margin-bottom: 1rem; }
    .card h3 { color: #38bdf8; margin-top: 0; }
    .card p { color: #94a3b8; }

    /* Risk badges */
    .risk-low { color: #4ade80; font-weight: bold; }
    .risk-medium { color: #fbbf24; font-weight: bold; }
    .risk-high { color: #f97316; font-weight: bold; }
    .risk-critical { color: #ef4444; font-weight: bold; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b; border-radius: 8px; padding: 8px 16px;
        color: #94a3b8; border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0ea5e9 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Init
# =============================================================================
if "auth_manager" not in st.session_state:
    st.session_state.auth_manager = AuthManager()
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "session_data" not in st.session_state:
    st.session_state.session_data = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "data" not in st.session_state:
    st.session_state.data = {}


# =============================================================================
# Auth Functions
# =============================================================================
def login_page():
    """Render the login page."""
    st.markdown("## 🔗 SCIRO")
    st.markdown("### Supply Chain Intelligence & Revenue Optimization")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### 🔐 Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", type="primary", use_container_width=True):
            token = st.session_state.auth_manager.authenticate(username, password)
            if token:
                st.session_state.authenticated = True
                st.session_state.auth_token = token
                st.session_state.session_data = st.session_state.auth_manager.validate_session(token)
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin/admin123, analyst/analyst123, or viewer/viewer123")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#64748b;font-size:0.8rem;'>"
            "Demo accounts: admin / analyst / viewer (password: [role]123)"
            "</div>",
            unsafe_allow_html=True,
        )


def check_permission(permission: str) -> bool:
    """Check if current user has a permission."""
    if not st.session_state.auth_token:
        return False
    return st.session_state.auth_manager.check_permission(
        st.session_state.auth_token, permission
    )


# =============================================================================
# Data Loading
# =============================================================================
def load_all_data(data_dir: str) -> dict:
    """Load all CSV data files."""
    data = {}
    for name in ("regions", "suppliers", "products", "inventory",
                  "shipments", "customers", "orders", "order_items"):
        path = os.path.join(data_dir, f"{name}.csv")
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            if name == "orders" and "order_date" in data[name].columns:
                data[name]["order_date"] = pd.to_datetime(
                    data[name]["order_date"], errors="coerce"
                )
            for col in ("shipped_date", "expected_delivery_date", "actual_delivery_date"):
                if name == "shipments" and col in data[name].columns:
                    data[name][col] = pd.to_datetime(data[name][col], errors="coerce")
    return data


def ensure_full_data_context() -> None:
    """Ensure baseline supply chain datasets exist in session state.

    If any standard table is missing, backfills it from sample data
    while preserving any uploaded custom datasets.
    """
    settings = get_settings()
    sample_dir = str(settings.SAMPLE_DATA_DIR)

    if "data" not in st.session_state:
        st.session_state.data = {}

    for name in ("regions", "suppliers", "products", "inventory",
                  "shipments", "customers", "orders", "order_items"):
        if name not in st.session_state.data or st.session_state.data[name] is None or getattr(st.session_state.data[name], "empty", True):
            path = os.path.join(sample_dir, f"{name}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if name == "orders" and "order_date" in df.columns:
                        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
                    for col in ("shipped_date", "expected_delivery_date", "actual_delivery_date"):
                        if name == "shipments" and col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                    st.session_state.data[name] = df
                except Exception as e:
                    pass
    st.session_state.data_loaded = True


def require_datasets(*keys: str) -> bool:
    """Ensure the requested datasets exist and are non-empty in session_state.data.

    Attempts to backfill missing tables using ensure_full_data_context().
    Returns False and renders a friendly warning if any required dataset
    remains unavailable.
    """
    if not st.session_state.get("data"):
        ensure_full_data_context()

    data = st.session_state.data
    missing = [k for k in keys if k not in data or data[k] is None or getattr(data[k], "empty", True)]
    if missing:
        ensure_full_data_context()
        data = st.session_state.data
        still_missing = [k for k in keys if k not in data or data[k] is None or getattr(data[k], "empty", True)]
        if still_missing:
            st.warning(
                f"⚠️ This section requires the following dataset(s): "
                f"{', '.join([f'`{k}`' for k in still_missing])}. "
                "Please upload or load sample data from the sidebar."
            )
            return False
    return True


def data_sidebar():
    """Data loading sidebar section."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Data Source")

    settings = get_settings()
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=str(settings.SAMPLE_DATA_DIR),
        key="data_dir_input",
    )

    if st.sidebar.button("🔄 Load Data", use_container_width=True):
        with st.spinner("Loading data..."):
            data = load_all_data(data_dir)
            if data:
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Loaded {len(data)} datasets")
            else:
                st.sidebar.error("❌ No data files found")

    if st.session_state.data_loaded:
        st.sidebar.markdown("**Loaded datasets:**")
        for name, df in st.session_state.data.items():
            st.sidebar.markdown(f"  • {name}: {len(df):,} rows")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Quick File Upload")
    quick_file = st.sidebar.file_uploader(
        "Upload PDF / Excel / CSV",
        type=["csv", "xlsx", "xls", "pdf"],
        key="sidebar_quick_file",
        help="Upload an Excel spreadsheet (.xlsx/.xls), PDF table/report (.pdf), or CSV file.",
    )
    if quick_file is not None:
        quick_entity = st.sidebar.selectbox(
            "Assign as Entity",
            options=["Auto-detect", "suppliers", "products", "inventory", "shipments", "customers", "orders", "order_items", "regions"],
            key="sidebar_quick_entity",
        )
        ext = detect_format(quick_file.name)
        sheet_choice = None
        if ext in (".xlsx", ".xls"):
            sheets = get_excel_sheet_names(quick_file)
            if len(sheets) > 1:
                sheet_choice = st.sidebar.selectbox("Sheet", sheets, key="sidebar_quick_sheet")
            elif sheets:
                sheet_choice = sheets[0]

        if st.sidebar.button("📥 Import & Apply", use_container_width=True, key="sidebar_quick_btn"):
            try:
                df = load_file(quick_file, sheet_name=sheet_choice, normalize=True)
                if df.empty:
                    st.sidebar.error("File is empty or no data detected.")
                else:
                    target = quick_entity.lower()
                    if target == "auto-detect":
                        fname = quick_file.name.lower()
                        sname = (sheet_choice or "").lower()
                        candidates = ["suppliers", "products", "inventory", "shipments", "customers", "orders", "order_items", "regions"]
                        inferred = None
                        for c in candidates:
                            if c in fname or c in sname:
                                inferred = c
                                break
                        target = inferred or os.path.splitext(quick_file.name)[0].lower().replace(" ", "_")

                    # Backfill any missing baseline tables so single-file upload doesn't break other analytics
                    ensure_full_data_context()
                    st.session_state.data[target] = df
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ Loaded '{target}' ({len(df):,} rows)")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Upload error: {e}")


# =============================================================================
# Dashboard Pages
# =============================================================================
def page_executive_dashboard():
    """Executive dashboard with KPI overview."""
    st.markdown("## 📊 Executive Dashboard")

    if not st.session_state.data_loaded:
        st.info("👆 Please load data using the sidebar to see the dashboard.")
        return

    # Ensure baseline data context is populated if user uploaded a single dataset
    ensure_full_data_context()
    data = st.session_state.data

    # Revenue KPIs
    if require_datasets("orders", "order_items", "products"):
        try:
            from sciro.analytics.revenue import get_revenue_summary, get_revenue_growth
            rev = get_revenue_summary(data["orders"], data["order_items"], data["products"], days=30)
            growth = get_revenue_growth(data["orders"], period_days=30)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Revenue (30d)", f"₹{rev['total_revenue']:,.0f}",
                        delta=f"{growth['growth_pct']:+.1f}%")
            col2.metric("Orders (30d)", f"{rev['total_orders']:,}",
                        delta=f"₹{rev['avg_order_value']:,.0f} avg")
            col3.metric("Gross Margin", f"{rev['gross_margin_pct']:.1f}%")
            col4.metric("Units Sold (30d)", f"{rev['total_units_sold']:,}")
        except Exception as e:
            st.warning(f"⚠️ Could not compute Revenue KPIs with active data: {e}")

    st.markdown("---")

    # Supply chain & inventory KPIs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔗 Supply Chain Health")
        if require_datasets("suppliers", "shipments", "products"):
            try:
                from sciro.analytics.supply_chain import get_supply_chain_summary
                sc = get_supply_chain_summary(data["suppliers"], data["shipments"], data["products"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Active Suppliers", sc["active_suppliers"])
                c2.metric("On-Time Delivery", f"{sc['avg_on_time_delivery_pct']}%")
                c3.metric("High Risk", sc["high_risk_supplier_count"],
                           delta=f"{sc['critical_risk_supplier_count']} critical", delta_color="inverse")
            except Exception as e:
                st.warning(f"⚠️ Could not compute Supply Chain metrics: {e}")

    with col2:
        st.markdown("### 📦 Inventory Health")
        if require_datasets("inventory", "products", "orders", "order_items"):
            try:
                from sciro.analytics.inventory import get_inventory_summary
                inv = get_inventory_summary(
                    data["inventory"], data["products"], data["orders"],
                    data["order_items"], data.get("suppliers"),
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Inventory Value", f"₹{inv['total_inventory_value']:,.0f}")
                c2.metric("Out of Stock", inv["out_of_stock_count"],
                           delta=f"{inv['critical_stock_count']} critical", delta_color="inverse")
                c3.metric("High Stockout Risk", inv["high_stockout_risk_count"])
            except Exception as e:
                st.warning(f"⚠️ Could not compute Inventory metrics: {e}")

    st.markdown("---")

    # Revenue trend chart
    st.markdown("### 📈 Revenue Trend")
    if require_datasets("orders", "order_items"):
        try:
            from sciro.analytics.revenue import get_monthly_revenue_trend
            trend = get_monthly_revenue_trend(data["orders"], data["order_items"], months=12)
            if not trend.empty:
                st.line_chart(trend.set_index("month")["revenue"])
        except Exception as e:
            st.warning(f"⚠️ Could not generate Revenue Trend: {e}")

    # Risk overview
    st.markdown("### ⚠️ Revenue-at-Risk Summary")
    if require_datasets("suppliers", "products", "inventory", "orders", "order_items", "shipments"):
        try:
            from sciro.analytics.risk_engine import get_risk_dashboard_data
            risk = get_risk_dashboard_data(
                data["suppliers"], data["products"], data["inventory"],
                data["orders"], data["order_items"], data["shipments"],
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Revenue at Risk", f"₹{risk['total_revenue_at_risk']:,.0f}")
            c2.metric("Products at Risk", risk["total_products_at_risk"])
            c3.metric("Affected Customers", risk["total_affected_customers"])
        except Exception as e:
            st.warning(f"⚠️ Could not compute Revenue-at-Risk summary: {e}")


def page_data_ingestion():
    """Data Upload & Ingestion page supporting PDF, Excel, and CSV."""
    st.markdown("## 📁 Data Upload & Ingestion")
    st.markdown("Upload supply chain records from **Excel spreadsheets (`.xlsx`, `.xls`)**, **PDF documents & invoices (`.pdf`)**, or **CSV files (`.csv`)**.")
    st.markdown("---")

    tab_upload, tab_active, tab_templates = st.tabs(["📤 Upload & Inspect File", "📊 Active Datasets", "📋 Schema Templates"])

    with tab_upload:
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Choose an Excel sheet, PDF report, or CSV file",
                type=["xlsx", "xls", "pdf", "csv"],
                key="main_data_uploader",
                help="Supports Excel workbooks (with sheet selection), PDF tables (via pdfplumber), and CSV files.",
            )
        with col_up2:
            entity_choice = st.selectbox(
                "Target Entity Mapping",
                options=[
                    "Auto-detect from filename/sheet",
                    "suppliers",
                    "products",
                    "inventory",
                    "shipments",
                    "customers",
                    "orders",
                    "order_items",
                    "regions",
                    "custom",
                ],
                key="main_entity_choice",
                help="Select which table this data should populate or overwrite in SCIRO.",
            )
            custom_entity_name = ""
            if entity_choice == "custom":
                custom_entity_name = st.text_input("Custom Entity Name", value="custom_data")

        if uploaded_file is not None:
            ext = detect_format(uploaded_file.name)
            file_size_kb = len(uploaded_file.getvalue()) / 1024
            st.info(f"📄 **{uploaded_file.name}** ({ext.upper()}) — {file_size_kb:.1f} KB")

            chosen_sheet = None
            if ext in (".xlsx", ".xls"):
                sheets = get_excel_sheet_names(uploaded_file)
                if len(sheets) > 1:
                    chosen_sheet = st.selectbox("📑 Select Excel Sheet", sheets, key="excel_sheet_select")
                elif sheets:
                    chosen_sheet = sheets[0]

            with st.spinner("Extracting data..."):
                try:
                    df = load_file(uploaded_file, sheet_name=chosen_sheet, normalize=True)
                except Exception as err:
                    st.error(f"❌ Failed to parse {uploaded_file.name}: {err}")
                    return

            if df.empty:
                st.warning("⚠️ No tabular data could be extracted from this file.")
                return

            # Determine target entity name
            if entity_choice == "Auto-detect from filename/sheet":
                fname = uploaded_file.name.lower()
                sname = (chosen_sheet or "").lower()
                candidates = ["suppliers", "products", "inventory", "shipments", "customers", "orders", "order_items", "regions"]
                target_entity = None
                for c in candidates:
                    if c in fname or c in sname:
                        target_entity = c
                        break
                target_entity = target_entity or os.path.splitext(uploaded_file.name)[0].lower().replace(" ", "_")
            elif entity_choice == "custom":
                target_entity = custom_entity_name.strip() or "custom_data"
            else:
                target_entity = entity_choice

            st.markdown(f"#### 🔍 Extracted Data Preview (`{target_entity}`: {len(df):,} rows × {len(df.columns)} columns)")
            st.dataframe(df.head(50), use_container_width=True)

            # Validation & ETL configuration
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                run_val = st.checkbox("Run Pydantic schema validation", value=True, help="Validates rows against SCIRO data models")
            with col_v2:
                apply_clean = st.checkbox("Apply ETL auto-cleaning", value=True, help="Trims strings, coerces numbers/dates, and removes duplicate IDs")

            cleaned_df = df.copy()
            if apply_clean:
                try:
                    cleaned_df = clean_dataframe(cleaned_df)
                except Exception as e:
                    logger.warning("ETL clean notice: %s", e)

            if run_val and target_entity in ["suppliers", "products", "inventory", "shipments", "customers", "orders", "order_items", "regions"]:
                valid_records, errors = validate_dataframe(cleaned_df, target_entity)
                val_c1, val_c2 = st.columns(2)
                val_c1.metric("Valid Rows", len(valid_records))
                val_c2.metric("Validation Issues", len(errors), delta=None if len(errors) == 0 else "needs review", delta_color="inverse")

                if errors:
                    with st.expander(f"⚠️ View {len(errors)} Validation Error(s)", expanded=False):
                        st.dataframe(pd.DataFrame(errors), use_container_width=True)

            st.markdown("---")
            act_col1, act_col2 = st.columns([1, 1])

            with act_col1:
                if st.button(f"🚀 Load into SCIRO Active Data as '{target_entity}'", type="primary", use_container_width=True):
                    ensure_full_data_context()
                    st.session_state.data[target_entity] = cleaned_df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Successfully loaded {len(cleaned_df):,} records into dataset '{target_entity}'! All analytics & ML pages will use this data.")
                    st.rerun()

            with act_col2:
                if st.button("💾 Save Uploaded File to data/uploads/", use_container_width=True):
                    upload_dir = Path(_PROJECT_ROOT) / "data" / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = upload_dir / uploaded_file.name
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.success(f"✅ File saved to {dest_path}")

    with tab_active:
        st.markdown("### 📊 In-Memory Active Datasets")
        if not st.session_state.data_loaded or not st.session_state.data:
            st.info("No datasets loaded yet. Upload a file or load sample data from the sidebar.")
        else:
            for name, dframe in st.session_state.data.items():
                with st.expander(f"📁 {name.upper()} ({len(dframe):,} rows, {len(dframe.columns)} cols)", expanded=False):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.dataframe(dframe.head(20), use_container_width=True)
                    with c2:
                        st.markdown(f"**Columns:**\n" + "\n".join([f"- `{c}`" for c in dframe.columns[:10]]))
                        if len(dframe.columns) > 10:
                            st.caption(f"...and {len(dframe.columns)-10} more")
                        csv_data = dframe.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"⬇️ Export {name}.csv",
                            data=csv_data,
                            file_name=f"{name}.csv",
                            mime="text/csv",
                            key=f"export_{name}",
                        )

    with tab_templates:
        st.markdown("### 📋 SCIRO Entity Schema Specifications")
        st.markdown("Ensure your Excel sheets or PDFs contain the following column headers for automated validation:")

        schemas = {
            "suppliers": ["name", "contact_email", "contact_phone", "lead_time_days", "reliability_score", "status", "region_name"],
            "products": ["sku", "name", "category", "unit_price", "cost_price", "weight_kg", "reorder_point", "safety_stock", "status", "supplier_name"],
            "inventory": ["product_sku", "region_name", "quantity_on_hand", "quantity_reserved"],
            "shipments": ["tracking_number", "supplier_name", "product_sku", "region_name", "status", "carrier", "shipping_cost", "shipped_date", "expected_delivery_date", "actual_delivery_date"],
            "customers": ["name", "email", "region_name", "company_name", "tier", "shipping_address"],
            "orders": ["order_number", "customer_email", "region_name", "order_date", "status", "payment_status", "total_amount"],
            "order_items": ["order_number", "product_sku", "quantity", "unit_price", "discount"],
            "regions": ["name", "code", "country"],
        }
        for ent, cols in schemas.items():
            with st.expander(f"Entity: {ent.upper()}", expanded=False):
                st.markdown("**Expected Columns:** " + ", ".join([f"`{c}`" for c in cols]))
                sample_df = pd.DataFrame(columns=cols)
                csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {ent}_template.csv",
                    data=csv_bytes,
                    file_name=f"{ent}_template.csv",
                    mime="text/csv",
                    key=f"template_{ent}",
                )


def page_supply_chain():
    """Supply chain analytics page."""
    st.markdown("## 🔗 Supply Chain Analytics")
    if not st.session_state.data_loaded:
        st.info("Please load data first.")
        return

    ensure_full_data_context()
    if not require_datasets("suppliers", "shipments", "products"):
        return

    data = st.session_state.data
    from sciro.analytics.supply_chain import (
        get_supplier_performance, calculate_supplier_risk_scores,
        get_shipment_anomalies,
    )

    tab1, tab2, tab3 = st.tabs(["Supplier Performance", "Risk Scores", "Shipment Anomalies"])

    with tab1:
        try:
            perf = get_supplier_performance(data["suppliers"], data["shipments"], data["products"])
            st.dataframe(perf, use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Supplier Performance: {e}")

    with tab2:
        try:
            perf = get_supplier_performance(data["suppliers"], data["shipments"], data["products"])
            scored = calculate_supplier_risk_scores(perf)
            display_cols = ["supplier_name", "risk_score", "risk_level", "on_time_delivery_pct",
                            "avg_delay_days", "products_supplied", "total_shipping_cost"]
            available_cols = [c for c in display_cols if c in scored.columns]
            st.dataframe(scored[available_cols], use_container_width=True, height=400)

            # Risk distribution chart
            dist = scored["risk_level"].value_counts()
            st.bar_chart(dist)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Supplier Risk Scores: {e}")

    with tab3:
        try:
            anomalies = get_shipment_anomalies(data["shipments"])
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True, height=400)
            else:
                st.success("No shipment anomalies detected.")
        except Exception as e:
            st.warning(f"⚠️ Could not analyze Shipment Anomalies: {e}")


def page_inventory():
    """Inventory analytics page."""
    st.markdown("## 📦 Inventory Analytics")
    if not st.session_state.data_loaded:
        st.info("Please load data first.")
        return

    ensure_full_data_context()
    if not require_datasets("inventory", "products", "orders", "order_items"):
        return

    data = st.session_state.data
    from sciro.analytics.inventory import (
        get_inventory_status, calculate_stockout_probability, calculate_inventory_turnover,
    )

    tab1, tab2, tab3 = st.tabs(["Stock Status", "Stockout Risk", "Turnover"])

    with tab1:
        try:
            status = get_inventory_status(
                data["inventory"], data["products"], data["orders"],
                data["order_items"], data.get("suppliers"),
            )
            # Summary counts
            counts = status["stock_status"].value_counts()
            st.bar_chart(counts)
            st.dataframe(status, use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"⚠️ Could not evaluate Stock Status: {e}")

    with tab2:
        try:
            status = get_inventory_status(
                data["inventory"], data["products"], data["orders"],
                data["order_items"], data.get("suppliers"),
            )
            stockout = calculate_stockout_probability(status)
            high_risk = stockout[stockout["stockout_risk"].isin(["HIGH", "CRITICAL"])]
            st.metric("Products at High Stockout Risk", len(high_risk))
            if not high_risk.empty:
                display_cols = ["product_sku", "name", "region_name", "quantity_available",
                                "days_of_supply", "stockout_probability", "stockout_risk"]
                available = [c for c in display_cols if c in high_risk.columns]
                st.dataframe(high_risk[available], use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not calculate Stockout Risk: {e}")

    with tab3:
        try:
            turnover = calculate_inventory_turnover(
                data["inventory"], data["products"], data["orders"], data["order_items"],
            )
            if not turnover.empty:
                st.dataframe(turnover, use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"⚠️ Could not calculate Inventory Turnover: {e}")


def page_revenue():
    """Revenue analytics page."""
    st.markdown("## 💰 Revenue Analytics")
    if not st.session_state.data_loaded:
        st.info("Please load data first.")
        return

    ensure_full_data_context()
    if not require_datasets("orders", "order_items", "products"):
        return

    data = st.session_state.data
    from sciro.analytics.revenue import (
        get_revenue_summary, get_revenue_by_product,
        get_revenue_by_region, get_monthly_revenue_trend,
    )

    tab1, tab2, tab3 = st.tabs(["Overview", "By Product", "By Region"])

    with tab1:
        try:
            days = st.slider("Period (days)", 7, 365, 30)
            rev = get_revenue_summary(data["orders"], data["order_items"], data["products"], days=days)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"₹{rev['total_revenue']:,.0f}")
            c2.metric("Orders", rev["total_orders"])
            c3.metric("Gross Profit", f"₹{rev['gross_profit']:,.0f}")
            c4.metric("Margin", f"{rev['gross_margin_pct']:.1f}%")

            trend = get_monthly_revenue_trend(data["orders"], data["order_items"], months=12)
            if not trend.empty:
                st.line_chart(trend.set_index("month")[["revenue", "orders"]])
        except Exception as e:
            st.warning(f"⚠️ Could not compute Revenue Overview: {e}")

    with tab2:
        try:
            products = get_revenue_by_product(
                data["orders"], data["order_items"], data["products"], days=90,
            )
            if not products.empty:
                st.bar_chart(products.set_index("product_name")["revenue"].head(15))
                st.dataframe(products, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Product Revenue: {e}")

    with tab3:
        try:
            regions = get_revenue_by_region(data["orders"], data["order_items"], days=90)
            if not regions.empty:
                st.bar_chart(regions.set_index("region_name")["revenue"])
                st.dataframe(regions, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Regional Revenue: {e}")


def page_customers():
    """Customer analytics page."""
    st.markdown("## 👥 Customer Analytics")
    if not st.session_state.data_loaded:
        st.info("Please load data first.")
        return

    ensure_full_data_context()
    if not require_datasets("customers", "orders", "order_items"):
        return

    data = st.session_state.data
    from sciro.analytics.customer import (
        get_customer_summary, segment_customers,
        calculate_customer_ltv, calculate_churn_probability,
    )

    tab1, tab2, tab3 = st.tabs(["Overview", "Segmentation", "Churn Risk"])

    with tab1:
        try:
            summary = get_customer_summary(data["customers"], data["orders"], data["order_items"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers", summary["total_customers"])
            c2.metric("Active (90d)", summary["active_customers"])
            c3.metric("New (30d)", summary["new_customers"])
            c4.metric("At Risk", summary["at_risk_customers"])

            ltv = calculate_customer_ltv(data["customers"], data["orders"], data["order_items"])
            if not ltv.empty:
                st.markdown("### Customer LTV Distribution")
                st.bar_chart(ltv["clv_tier"].value_counts())
        except Exception as e:
            st.warning(f"⚠️ Could not compute Customer Summary: {e}")

    with tab2:
        try:
            segments = segment_customers(data["customers"], data["orders"], data["order_items"])
            if not segments.empty:
                seg_counts = segments["segment"].value_counts()
                st.bar_chart(seg_counts)
                st.dataframe(segments, use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"⚠️ Could not segment Customers: {e}")

    with tab3:
        try:
            churn = calculate_churn_probability(data["customers"], data["orders"])
            if not churn.empty:
                risk_counts = churn["churn_risk"].value_counts()
                st.bar_chart(risk_counts)
                high_churn = churn[churn["churn_risk"] == "High"]
                if not high_churn.empty:
                    st.markdown(f"### {len(high_churn)} High Churn Risk Customers")
                    st.dataframe(high_churn, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not calculate Churn Risk: {e}")


def page_ml_models():
    """ML models page (Module 3)."""
    st.markdown("## 🤖 ML Prediction Engine")
    if not st.session_state.data_loaded:
        st.info("Please load data first.")
        return

    ensure_full_data_context()
    if not check_permission("run_ml_models"):
        st.warning("⛔ You don't have permission to run ML models.")
        return

    data = st.session_state.data

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demand Forecast", "Revenue Prediction", "Stockout Prediction", "Supplier Risk",
    ])

    with tab1:
        st.markdown("### 📈 Demand Forecaster")
        st.markdown("Predicts future product demand using gradient boosting with time-series features.")
        if st.button("Train & Predict Demand", key="btn_demand"):
            with st.spinner("Training demand forecaster..."):
                try:
                    from sciro.ml.demand_forecaster import DemandForecaster
                    forecaster = DemandForecaster()
                    features = forecaster.prepare_data(
                        data["orders"], data["order_items"], data["products"],
                    )
                    metrics = forecaster.train(features)
                    st.success(f"✅ Model trained — MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, R²: {metrics['r2']}")

                    forecast = forecaster.forecast_next_period(features)
                    st.markdown("### Next Period Forecast")
                    st.dataframe(forecast, use_container_width=True)

                    imp = forecaster.get_feature_importance()
                    st.markdown("### Feature Importance")
                    st.bar_chart(imp.set_index("feature")["importance"])
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.markdown("### 💰 Revenue Predictor")
        st.markdown("Predicts monthly revenue using lagged features and calendar indicators.")
        months_ahead = st.slider("Months to forecast", 1, 6, 3, key="rev_months")
        if st.button("Train & Forecast Revenue", key="btn_revenue"):
            with st.spinner("Training revenue predictor..."):
                try:
                    from sciro.ml.revenue_predictor import RevenuePredictor
                    predictor = RevenuePredictor()
                    features = predictor.prepare_data(
                        data["orders"], data["order_items"], data["products"],
                    )
                    metrics = predictor.train(features)
                    st.success(f"✅ Model trained — MAE: ₹{metrics['mae']:,.0f}, R²: {metrics['r2']}, MAPE: {metrics['mape']}%")

                    forecast = predictor.forecast_next_months(features, months_ahead=months_ahead)
                    st.markdown("### Revenue Forecast")
                    st.dataframe(forecast, use_container_width=True)
                    st.line_chart(forecast.set_index("month")["predicted_revenue"])
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.markdown("### 📦 Stockout Predictor")
        st.markdown("Identifies products at risk of stockout using Random Forest classification.")
        if st.button("Train & Predict Stockouts", key="btn_stockout"):
            with st.spinner("Training stockout predictor..."):
                try:
                    from sciro.ml.stockout_predictor import StockoutPredictor
                    predictor = StockoutPredictor()
                    features = predictor.prepare_data(
                        data["inventory"], data["products"], data["orders"],
                        data["order_items"], data["suppliers"], data["shipments"],
                    )
                    metrics = predictor.train(features)
                    st.success(f"✅ Model trained — Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}")

                    predictions = predictor.predict(features)
                    at_risk = predictor.get_at_risk_products(predictions)
                    st.markdown(f"### {len(at_risk)} Products at Predicted Stockout Risk")
                    if not at_risk.empty:
                        st.dataframe(at_risk, use_container_width=True)

                    imp = predictor.get_feature_importance()
                    st.markdown("### Feature Importance")
                    st.bar_chart(imp.set_index("feature")["importance"])
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab4:
        st.markdown("### ⚠️ Supplier Risk Classifier")
        st.markdown("Predicts supplier risk level (LOW/MEDIUM/HIGH/CRITICAL).")
        if st.button("Train & Classify", key="btn_supplier_risk"):
            with st.spinner("Training supplier risk model..."):
                try:
                    from sciro.ml.supplier_risk_classifier import SupplierRiskClassifier
                    classifier = SupplierRiskClassifier()
                    features = classifier.prepare_data(
                        data["suppliers"], data["shipments"], data["products"],
                        data["orders"], data["order_items"],
                    )
                    metrics = classifier.train(features)
                    st.success(f"✅ Model trained — Accuracy: {metrics['accuracy']}, F1: {metrics['f1_weighted']}")

                    predictions = classifier.predict(features)
                    high_risk = classifier.get_high_risk_suppliers(predictions)
                    st.markdown(f"### {len(high_risk)} High/Critical Risk Suppliers")
                    if not high_risk.empty:
                        st.dataframe(high_risk, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")


def page_simulation():
    """Simulation & optimization page (Module 4)."""
    st.markdown("## 🎲 Simulation & Optimization")
    if not st.session_state.data_loaded:
        st.info("Please load data first.")
        return

    ensure_full_data_context()
    if not check_permission("run_simulations"):
        st.warning("⛔ You don't have permission to run simulations.")
        return

    if not require_datasets("suppliers", "products", "inventory", "orders", "order_items"):
        return

    data = st.session_state.data

    tab1, tab2, tab3 = st.tabs(["Disruption Simulation", "Pricing Optimization", "Inventory Optimization"])

    with tab1:
        st.markdown("### 🌪️ Monte Carlo Disruption Simulation")
        col1, col2 = st.columns(2)

        with col1:
            supplier_list = data["suppliers"]["name"].tolist() if "suppliers" in data else []
            failed = st.multiselect("Suppliers to simulate failure", supplier_list)
            demand_surge = st.slider("Demand surge %", 0, 200, 0) / 100
            horizon = st.slider("Horizon (days)", 7, 90, 30)

        with col2:
            num_sims = st.slider("Number of simulations", 100, 5000, 1000, step=100)
            scenario_name = st.text_input("Scenario name", "Custom Scenario")

        if st.button("🚀 Run Simulation", key="btn_sim", type="primary"):
            with st.spinner(f"Running {num_sims} simulations..."):
                try:
                    from sciro.simulation.disruption_simulator import DisruptionSimulator, ScenarioConfig
                    sim = DisruptionSimulator()
                    config = ScenarioConfig(
                        name=scenario_name,
                        failed_suppliers=failed,
                        demand_surge_pct=demand_surge,
                        num_simulations=num_sims,
                        horizon_days=horizon,
                    )
                    result = sim.run_simulation(
                        config, data["products"], data["inventory"],
                        data["orders"], data["order_items"], data["suppliers"],
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean Revenue at Risk", f"₹{result.revenue_at_risk_mean:,.0f}")
                    c2.metric("P95 Revenue at Risk", f"₹{result.revenue_at_risk_p95:,.0f}")
                    c3.metric("Avg Products Affected", f"{result.affected_products_mean:.0f}")

                    st.markdown("### Distribution of Revenue Loss")
                    hist_data = pd.DataFrame({"revenue_loss": result.total_lost_revenue_distribution})
                    st.bar_chart(hist_data["revenue_loss"].value_counts().sort_index())

                    if result.product_impact:
                        st.markdown("### Most Vulnerable Products")
                        st.dataframe(pd.DataFrame(result.product_impact), use_container_width=True)

                    st.text(result.summary)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.markdown("### 💲 Pricing Optimization")
        objective = st.radio("Optimization objective", ["profit", "revenue"])
        price_range = st.slider("Price exploration range (%)", 5, 50, 30) / 100

        if st.button("🎯 Optimize Prices", key="btn_price"):
            with st.spinner("Optimizing prices..."):
                try:
                    from sciro.simulation.pricing_optimizer import PricingOptimizer
                    optimizer = PricingOptimizer()
                    elasticity = optimizer.estimate_elasticity(
                        data["orders"], data["order_items"], data["products"],
                    )
                    results = optimizer.optimize_prices(
                        data["products"], elasticity,
                        price_range_pct=price_range, objective=objective,
                    )

                    summary = optimizer.get_optimization_summary()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Products Analyzed", summary["products_analyzed"])
                    c2.metric("Current Total Profit", f"₹{summary['total_current_profit']:,.0f}")
                    c3.metric("Optimized Total Profit", f"₹{summary['total_predicted_profit']:,.0f}")

                    st.dataframe(results, use_container_width=True, height=400)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.markdown("### 📊 Inventory Optimization")

        sub_tab = st.radio("Analysis type", ["EOQ Analysis", "Regional Allocation"])

        if sub_tab == "EOQ Analysis":
            holding = st.slider("Holding cost (%)", 5, 40, 20) / 100
            ordering = st.number_input("Ordering cost (₹)", value=500, step=100)
            if st.button("Calculate EOQ", key="btn_eoq"):
                with st.spinner("Calculating..."):
                    try:
                        from sciro.simulation.inventory_optimizer import InventoryOptimizer
                        optimizer = InventoryOptimizer()
                        eoq = optimizer.calculate_eoq(
                            data["products"], data["orders"], data["order_items"],
                            holding_cost_pct=holding, ordering_cost=ordering,
                        )
                        st.dataframe(eoq, use_container_width=True, height=400)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            if st.button("Analyze Allocation", key="btn_alloc"):
                with st.spinner("Analyzing..."):
                    try:
                        from sciro.simulation.inventory_optimizer import InventoryOptimizer
                        optimizer = InventoryOptimizer()
                        alloc = optimizer.optimize_allocation(
                            data["inventory"], data["products"],
                            data["orders"], data["order_items"],
                        )
                        summary = optimizer.get_optimization_summary()
                        if "allocation" in summary:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Overstocked", summary["allocation"]["overstocked"])
                            c2.metric("Understocked", summary["allocation"]["understocked"])
                            c3.metric("Balanced", summary["allocation"]["balanced"])
                        st.dataframe(alloc, use_container_width=True, height=400)
                    except Exception as e:
                        st.error(f"Error: {e}")


def page_copilot():
    """AI Copilot page."""
    st.markdown("## 🤖 AI Copilot")
    st.markdown("Ask me anything about your supply chain, revenue, inventory, or customers.")

    if not st.session_state.data_loaded:
        st.info("Please load data first to use the copilot.")
        return

    ensure_full_data_context()

    # Initialize copilot
    if "copilot" not in st.session_state:
        from sciro.dashboard.copilot import AICopilot
        st.session_state.copilot = AICopilot()

    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    # Ensure copilot has data
    copilot = st.session_state.copilot
    copilot._data_cache = st.session_state.data

    # Query input
    query = st.text_input(
        "💬 Ask a question",
        placeholder='e.g. "What\'s our revenue for the last 30 days?"',
        key="copilot_input",
    )

    if query:
        with st.spinner("Thinking..."):
            result = copilot.query(query)

        st.session_state.copilot_history.append(result)

        st.markdown(result["response"])

        if result.get("data"):
            with st.expander("📊 Raw Data"):
                if isinstance(result["data"], dict):
                    st.json(result["data"])
                elif isinstance(result["data"], list):
                    st.dataframe(pd.DataFrame(result["data"]))

    # Chat history
    if st.session_state.copilot_history:
        st.markdown("---")
        st.markdown("### 📜 Query History")
        for i, entry in enumerate(reversed(st.session_state.copilot_history[-10:])):
            with st.expander(f"Q: {entry['query'][:80]}...", expanded=False):
                st.markdown(f"**Intent:** {entry['intent']} (confidence: {entry['confidence']})")
                st.markdown(entry["response"])


def page_settings():
    """Settings & user management (admin only)."""
    st.markdown("## ⚙️ Settings")

    if not check_permission("manage_users"):
        st.warning("⛔ Only admins can access settings.")
        return

    tab1, tab2 = st.tabs(["User Management", "System Info"])

    with tab1:
        st.markdown("### 👥 Users")
        users = st.session_state.auth_manager.list_users()
        st.dataframe(pd.DataFrame(users), use_container_width=True)

        st.markdown("### ➕ Add User")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_user = st.text_input("Username", key="new_username")
        with col2:
            new_pass = st.text_input("Password", type="password", key="new_password")
        with col3:
            new_role = st.selectbox("Role", ["viewer", "analyst", "admin"])

        if st.button("Create User"):
            if new_user and new_pass:
                if st.session_state.auth_manager.create_user(new_user, new_pass, new_role):
                    st.success(f"✅ User '{new_user}' created")
                else:
                    st.error("User already exists or invalid role")

    with tab2:
        st.markdown("### System Information")
        settings = get_settings()
        st.json({
            "project_root": str(settings.PROJECT_ROOT),
            "data_dir": str(settings.DATA_DIR),
            "db_host": settings.db.host,
            "db_name": settings.db.name,
            "log_level": settings.LOG_LEVEL,
            "data_loaded": st.session_state.data_loaded,
            "datasets": list(st.session_state.data.keys()) if st.session_state.data else [],
        })


# =============================================================================
# Main App
# =============================================================================
def main():
    """Main application entry point."""

    # Authentication gate
    if not st.session_state.authenticated:
        login_page()
        return

    # Sidebar navigation
    session = st.session_state.session_data
    st.sidebar.markdown(f"### 🔗 SCIRO")
    st.sidebar.markdown(f"👤 **{session['full_name']}** ({session['role']})")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.auth_manager.logout(st.session_state.auth_token)
        st.session_state.authenticated = False
        st.session_state.auth_token = None
        st.session_state.session_data = None
        st.rerun()

    data_sidebar()
    st.sidebar.markdown("---")

    # Navigation
    pages = {
        "📊 Dashboard": page_executive_dashboard,
        "📁 Data Upload & Ingestion": page_data_ingestion,
        "🔗 Supply Chain": page_supply_chain,
        "📦 Inventory": page_inventory,
        "💰 Revenue": page_revenue,
        "👥 Customers": page_customers,
        "🤖 ML Models": page_ml_models,
        "🎲 Simulation": page_simulation,
        "💬 AI Copilot": page_copilot,
        "⚙️ Settings": page_settings,
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    pages[selection]()


if __name__ == "__main__":
    main()
