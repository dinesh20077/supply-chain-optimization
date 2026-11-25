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
from sciro.dashboard.charts import (
    create_revenue_trend_chart,
    create_revenue_at_risk_donut,
    create_product_revenue_bar,
    create_regional_revenue_chart,
    create_supplier_risk_distribution,
    create_supplier_scatter,
    create_inventory_health_donut,
    create_stockout_risk_scatter,
    create_customer_segment_treemap,
    create_customer_ltv_bar,
    create_feature_importance_chart,
    create_revenue_forecast_chart,
    create_monte_carlo_histogram,
)

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
# =============================================================================
# Custom CSS — Enterprise Executive Theme & Slate Layering
# =============================================================================
st.markdown("""
<style>
    /* Font family & layout */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Hide Streamlit Header Chrome, Deploy Button & Menus */
    #MainMenu { visibility: hidden !important; display: none !important; }
    header[data-testid="stHeader"] { visibility: hidden !important; display: none !important; }
    footer { visibility: hidden !important; display: none !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
    [data-testid="stDecoration"] { visibility: hidden !important; display: none !important; }
    [data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }
    .stDeployButton { display: none !important; }

    /* Layered Slate Backgrounds & Layout */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }
    .main .block-container { 
        padding-top: 1.5rem; 
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }

    /* Sidebar Styling — Deep Midnight Slate */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #111827 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Gradient Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    h2 {
        background: linear-gradient(135deg, #f8fafc 0%, #cbd5e1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        color: #e2e8f0 !important;
        font-size: 1.15rem !important;
        margin-top: 1rem !important;
    }

    /* Enterprise Glassmorphic KPI Cards */
    [data-testid="stMetric"] {
        background: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 2px solid #4f46e5 !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(79, 70, 229, 0.45) !important;
        box-shadow: 0 8px 24px -4px rgba(79, 70, 229, 0.25) !important;
    }
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    /* Skeleton / Ghosted KPI Cards */
    .skeleton-kpi {
        background: #1e293b;
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-top: 2px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        opacity: 0.75;
        transition: all 0.2s ease;
    }
    .skeleton-kpi:hover {
        opacity: 0.9;
        border-color: rgba(255, 255, 255, 0.2);
    }
    .skeleton-kpi-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .skeleton-kpi-value {
        color: #94a3b8;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 4px 0;
        letter-spacing: -0.02em;
    }
    .skeleton-kpi-sub {
        color: #475569;
        font-size: 0.8rem;
    }

    /* Hero Onboarding Card */
    .hero-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid #4f46e5;
        border-radius: 18px;
        padding: 2.2rem 2rem;
        max-width: 680px;
        margin: 1.5rem auto 1rem auto;
        text-align: center;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(12px);
    }
    .hero-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: rgba(79, 70, 229, 0.15);
        color: #818cf8;
        border: 1px solid rgba(79, 70, 229, 0.3);
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.55;
        max-width: 560px;
        margin: 0 auto 1.6rem auto;
    }
    .steps-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(11, 15, 25, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1.6rem;
    }
    .step-item {
        flex: 1;
        text-align: center;
    }
    .step-badge {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 24px;
        border-radius: 50%;
        background: rgba(79, 70, 229, 0.25);
        color: #a5b4fc;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 5px;
        border: 1px solid rgba(79, 70, 229, 0.4);
    }
    .step-title {
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .step-desc {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 2px;
    }
    .step-arrow {
        color: #475569;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0 4px;
    }

    /* Buttons with Authoritative Enterprise Accent (Electric Indigo & Cyan) */
    .stButton > button[kind="primary"], .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 0.52rem 1.25rem !important;
        box-shadow: 0 4px 14px -2px rgba(79, 70, 229, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #0284c7 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px -2px rgba(79, 70, 229, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 6px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9 !important;
        border-color: rgba(255, 255, 255, 0.16) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.25) 0%, rgba(14, 165, 233, 0.25) 100%) !important;
        border-color: #4f46e5 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 0 14px rgba(79, 70, 229, 0.25) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-danger { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-info { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); }
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
    """Render the executive login portal."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1.5rem 0;'>
        <div style='font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            SCIRO Enterprise
        </div>
        <div style='color: #94a3b8; font-size: 1rem; margin-top: 0.35rem;'>
            Supply Chain Intelligence & Revenue Optimization Decision Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #131b2e 0%, #1e293b 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 16px; padding: 2rem; box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);'>
            <h4 style='color: #f8fafc; margin-top:0; margin-bottom: 1.2rem; font-weight:600;'>🔐 Enterprise Authentication</h4>
        """, unsafe_allow_html=True)
        username = st.text_input("Username", key="login_user", placeholder="Enter username (e.g. admin)")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")

        if st.button("Sign In to SCIRO", type="primary", use_container_width=True):
            token = st.session_state.auth_manager.authenticate(username, password)
            if token:
                st.session_state.authenticated = True
                st.session_state.auth_token = token
                st.session_state.session_data = st.session_state.auth_manager.validate_session(token)
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin/admin123, analyst/analyst123, or viewer/viewer123")

        st.markdown("""
            <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #334155; text-align: center;'>
                <div style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.5rem;'>Available Demo Roles</div>
                <div style='display: flex; justify-content: center; gap: 8px;'>
                    <span class='badge badge-info'>admin : admin123</span>
                    <span class='badge badge-success'>analyst : analyst123</span>
                    <span class='badge badge-warning'>viewer : viewer123</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


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


def get_available_sample_datasets() -> dict:
    """Scan data directory for available sample dataset folders."""
    settings = get_settings()
    data_dir = Path(settings.DATA_DIR)
    found = {}
    if settings.SAMPLE_DATA_DIR.exists():
        found["Enterprise Demo (Default)"] = str(settings.SAMPLE_DATA_DIR)

    if data_dir.exists():
        for item in sorted(data_dir.iterdir()):
            if item.is_dir() and item.name not in ("raw", "processed", "sample", ".git"):
                csvs = list(item.glob("*.csv"))
                if csvs:
                    display_name = item.name.replace("_", " ").title()
                    found[f"Scenario: {display_name} ({len(csvs)} files)"] = str(item)
    return found


def ensure_full_data_context(sample_dir: str = None) -> None:
    """Ensure baseline supply chain datasets exist in session state.

    If any standard table is missing, backfills it from sample data
    while preserving any uploaded custom datasets.
    """
    settings = get_settings()
    if not sample_dir:
        sample_dir = st.session_state.get("active_sample_dir", str(settings.SAMPLE_DATA_DIR))

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


# =============================================================================
# Empty State & Header Components
# =============================================================================
def render_executive_header():
    """Render unified clean header for Executive Overview."""
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown("""
        <div style='margin-bottom: 1.2rem;'>
            <div style='font-size: 2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;'>
                Executive Overview
            </div>
            <div style='color: #94a3b8; font-size: 0.95rem; font-weight: 400; margin-top: 3px;'>
                Real-time supply chain intelligence & predictive risk analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_head2:
        status_text = "● System Live • Active Analytics" if st.session_state.data_loaded else "○ Standby • Ready to Ingest"
        status_cls = "badge-success" if st.session_state.data_loaded else "badge-info"
        st.markdown(f"""
        <div style='text-align: right; margin-top: 0.6rem;'>
            <span class='badge {status_cls}' style='font-size: 0.8rem; padding: 6px 14px; font-weight: 600;'>
                {status_text}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, pill_text: str = ""):
    """Render unified clean header for SCIRO dashboard pages."""
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"""
        <div style='margin-bottom: 1.2rem;'>
            <div style='font-size: 1.9rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;'>
                {title}
            </div>
            <div style='color: #94a3b8; font-size: 0.95rem; font-weight: 400; margin-top: 2px;'>
                {subtitle}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        if pill_text:
            st.markdown(f"""
            <div style='text-align: right; margin-top: 0.6rem;'>
                <span class='badge badge-info' style='font-size: 0.8rem; padding: 6px 14px; font-weight: 600;'>
                    {pill_text}
                </span>
            </div>
            """, unsafe_allow_html=True)


def render_skeleton_kpis():
    """Display ghosted/skeleton KPI cards across the top when no data is loaded."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='skeleton-kpi'>
            <div class='skeleton-kpi-label'>Total Revenue (30d)</div>
            <div class='skeleton-kpi-value'>₹--</div>
            <div class='skeleton-kpi-sub'>-- vs previous period</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='skeleton-kpi'>
            <div class='skeleton-kpi-label'>Total Inventory Value</div>
            <div class='skeleton-kpi-value'>₹--</div>
            <div class='skeleton-kpi-sub'>-- active SKUs tracked</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='skeleton-kpi'>
            <div class='skeleton-kpi-label'>On-Time Delivery Rate</div>
            <div class='skeleton-kpi-value'>--%</div>
            <div class='skeleton-kpi-sub'>-- avg transit delay</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='skeleton-kpi'>
            <div class='skeleton-kpi-label'>Supply Risk Index</div>
            <div class='skeleton-kpi-value'>-- / 100</div>
            <div class='skeleton-kpi-sub'>-- suppliers monitored</div>
        </div>
        """, unsafe_allow_html=True)


def render_hero_onboarding():
    """Render centered glassmorphic hero onboarding card with 3-step visual guide and 1-click demo button."""
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown("""
        <div class='hero-card'>
            <div class='hero-badge'>SCIRO Enterprise Intelligence</div>
            <div class='hero-title'>Welcome to SCIRO Platform</div>
            <div class='hero-subtitle'>
                Autonomous multi-tier supply chain orchestration, inventory optimization, and predictive revenue-at-risk analytics powered by enterprise ML.
            </div>
            <div class='steps-container'>
                <div class='step-item'>
                    <div class='step-badge'>1</div>
                    <div class='step-title'>Connect Source</div>
                    <div class='step-desc'>CSV, Excel, PDF, ERP</div>
                </div>
                <div class='step-arrow'>→</div>
                <div class='step-item'>
                    <div class='step-badge'>2</div>
                    <div class='step-title'>AI Processing</div>
                    <div class='step-desc'>ETL, ML Risk & Evals</div>
                </div>
                <div class='step-arrow'>→</div>
                <div class='step-item'>
                    <div class='step-badge'>3</div>
                    <div class='step-title'>Live Insights</div>
                    <div class='step-desc'>KPIs & What-If Matrix</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        btn_c1, btn_c2, btn_c3 = st.columns([1, 3, 1])
        with btn_c2:
            if st.button("✨ Explore with Demo Supply Chain Data", type="primary", use_container_width=True, key="hero_explore_demo_btn"):
                with st.spinner("Initializing SCIRO analytics engine & loading demo datasets..."):
                    ensure_full_data_context()
                    st.session_state.data_loaded = True
                st.rerun()

        st.markdown("""
        <div style='text-align: center; margin-top: 0.8rem; color: #64748b; font-size: 0.8rem;'>
            Or select <strong>Upload Custom File</strong> from the Data Source sidebar menu.
        </div>
        """, unsafe_allow_html=True)


def render_generic_empty_state(page_title: str):
    """Render clean empty-state card for analytical sub-pages."""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(f"""
        <div class='hero-card' style='max-width: 580px; margin: 2rem auto 1rem auto; padding: 2rem 1.8rem;'>
            <div class='hero-badge'>Demo Dataset Available</div>
            <div class='hero-title' style='font-size: 1.45rem;'>Data Required for {page_title}</div>
            <div class='hero-subtitle' style='font-size: 0.88rem; margin-bottom: 1.4rem;'>
                This module cross-references your multi-tier supply chain records to compute live indicators and ML predictions. Load demo data to explore immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)

        btn_c1, btn_c2, btn_c3 = st.columns([1, 2.5, 1])
        with btn_c2:
            if st.button(f"⚡ Load Demo Data for {page_title}", type="primary", use_container_width=True, key=f"btn_demo_{page_title.lower().replace(' ', '_')}"):
                with st.spinner("Loading demo datasets..."):
                    ensure_full_data_context()
                    st.session_state.data_loaded = True
                st.rerun()


def data_sidebar():
    """Data loading sidebar section with dropdown source selector and hidden dev paths."""
    st.sidebar.markdown("### 🔌 Data Source")

    source_choice = st.sidebar.selectbox(
        "Source Mode",
        options=[
            "Enterprise Demo Warehouse (Built-in)",
            "Upload Custom File (CSV, Excel, PDF)",
            "Cloud ERP Connector (SAP / NetSuite)",
        ],
        key="sidebar_source_mode",
        label_visibility="collapsed",
    )

    if source_choice == "Enterprise Demo Warehouse (Built-in)":
        datasets = get_available_sample_datasets()
        chosen_label = None
        if len(datasets) > 1:
            chosen_label = st.sidebar.selectbox(
                "Select Scenario Dataset",
                options=list(datasets.keys()),
                key="sidebar_dataset_choice",
                help="Choose between available sample supply chain scenarios.",
            )
            target_dir = datasets[chosen_label]
        elif datasets:
            target_dir = list(datasets.values())[0]
            chosen_label = list(datasets.keys())[0]
        else:
            settings = get_settings()
            target_dir = str(settings.SAMPLE_DATA_DIR)

        current_active = st.session_state.get("active_sample_dir")
        if not st.session_state.data_loaded or (current_active and current_active != target_dir):
            btn_text = f"⚡ Load {chosen_label.split('(')[0].strip()}" if chosen_label else "⚡ Connect & Load Demo Data"
            if st.sidebar.button(btn_text, use_container_width=True, key="btn_load_demo_sb"):
                with st.spinner("Loading demo records..."):
                    st.session_state.active_sample_dir = target_dir
                    loaded = load_all_data(target_dir)
                    if loaded:
                        st.session_state.data = loaded
                        st.session_state.data_loaded = True
                    else:
                        ensure_full_data_context(target_dir)
                st.sidebar.success(f"✅ Loaded: {chosen_label or 'Demo Data'}")
                st.rerun()
        else:
            active_name = chosen_label or "Default Demo Warehouse"
            st.sidebar.markdown(f"""
            <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 6px 10px; margin-bottom: 8px; font-size: 0.8rem; color: #34d399;'>
                ● Active: {active_name}
            </div>
            """, unsafe_allow_html=True)
            if st.sidebar.button("🔄 Reload Demo Data", use_container_width=True, key="btn_reload_demo_sb"):
                with st.spinner("Refreshing data..."):
                    loaded = load_all_data(target_dir)
                    if loaded:
                        st.session_state.data = loaded
                        st.session_state.data_loaded = True
                    else:
                        ensure_full_data_context(target_dir)
                st.sidebar.success("✅ Refreshed")
                st.rerun()

    elif source_choice == "Upload Custom File (CSV, Excel, PDF)":
        quick_file = st.sidebar.file_uploader(
            "Upload File",
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

                        ensure_full_data_context()
                        st.session_state.data[target] = df
                        st.session_state.data_loaded = True
                        st.sidebar.success(f"✅ Loaded '{target}' ({len(df):,} rows)")
                        st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Upload error: {e}")

    elif source_choice == "Cloud ERP Connector (SAP / NetSuite)":
        st.sidebar.markdown("""
        <div style='background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.25); border-radius: 8px; padding: 8px; font-size: 0.8rem; color: #38bdf8;'>
            <strong>Cloud ERP Connector:</strong> Configured for SAP S/4HANA & NetSuite SuiteTalk API.
        </div>
        """, unsafe_allow_html=True)
        if st.sidebar.button("⚡ Sync Cloud ERP Data", use_container_width=True, key="btn_sync_erp"):
            with st.spinner("Syncing simulated ERP cloud records..."):
                ensure_full_data_context()
                st.session_state.data_loaded = True
            st.sidebar.success("✅ Synced ERP database")
            st.rerun()

    if st.session_state.data_loaded and st.session_state.get("data"):
        with st.sidebar.expander("📊 Loaded Entities Summary", expanded=False):
            for name, df in list(st.session_state.data.items()):
                if isinstance(df, tuple):
                    df = df[0]
                    st.session_state.data[name] = df
                if hasattr(df, "__len__"):
                    st.markdown(f"<span style='color:#94a3b8; font-size: 0.82rem;'>• {name}: <strong style='color:#e2e8f0;'>{len(df):,}</strong> rows</span>", unsafe_allow_html=True)

    # Hide developer local path behind an Advanced Settings expander
    settings = get_settings()
    with st.sidebar.expander("⚙️ Advanced Path Settings", expanded=False):
        data_dir = st.text_input(
            "Local Directory Path",
            value=str(settings.SAMPLE_DATA_DIR),
            key="data_dir_input",
            help="Specify a local filesystem folder holding SCIRO CSV files.",
        )
        if st.button("Load from Local Path", key="btn_load_custom_path", use_container_width=True):
            with st.spinner("Loading from directory..."):
                data = load_all_data(data_dir)
                if data:
                    st.session_state.data = data
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(data)} datasets")
                    st.rerun()
                else:
                    st.error("❌ No data files found in path")


# =============================================================================
# Dashboard Pages
# =============================================================================
def page_executive_dashboard():
    """Executive dashboard with KPI overview."""
    render_executive_header()

    if not st.session_state.data_loaded:
        render_skeleton_kpis()
        render_hero_onboarding()
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

    # Visual Analytics Row
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown("### 📈 Revenue Trajectory & Volume")
        if require_datasets("orders", "order_items"):
            try:
                from sciro.analytics.revenue import get_monthly_revenue_trend
                trend = get_monthly_revenue_trend(data["orders"], data["order_items"], months=12)
                if not trend.empty:
                    fig_trend = create_revenue_trend_chart(trend)
                    st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not generate Revenue Trend: {e}")

    with col_chart2:
        st.markdown("### ⚠️ Revenue-at-Risk Breakdown")
        if require_datasets("suppliers", "products", "inventory", "orders", "order_items", "shipments"):
            try:
                from sciro.analytics.risk_engine import get_risk_dashboard_data
                risk = get_risk_dashboard_data(
                    data["suppliers"], data["products"], data["inventory"],
                    data["orders"], data["order_items"], data["shipments"],
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Revenue at Risk", f"₹{risk['total_revenue_at_risk']:,.0f}")
                c2.metric("Products at Risk", risk["total_products_at_risk"])
                c3.metric("Affected Customers", risk["total_affected_customers"])

                fig_risk = create_revenue_at_risk_donut(risk)
                st.plotly_chart(fig_risk, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not compute Revenue-at-Risk summary: {e}")


def page_data_ingestion():
    """Data Upload & Ingestion page supporting PDF, Excel, and CSV."""
    render_page_header(
        "Data Ingestion & Pipeline",
        "Enterprise multi-format data intake supporting Excel (.xlsx, .xls), PDF invoices/tables (.pdf), and CSV (.csv)",
        "● ETL Ready",
    )
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
                    clean_res = clean_dataframe(cleaned_df)
                    if isinstance(clean_res, tuple):
                        cleaned_df, clean_report = clean_res
                    else:
                        cleaned_df = clean_res
                except Exception as e:
                    pass

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
                    if isinstance(cleaned_df, tuple):
                        cleaned_df = cleaned_df[0]
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
            for name, dframe in list(st.session_state.data.items()):
                if isinstance(dframe, tuple):
                    dframe = dframe[0]
                    st.session_state.data[name] = dframe
                if not hasattr(dframe, "columns"):
                    continue
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
    render_page_header(
        "Supply Chain Intelligence",
        "Multi-tier supplier performance, on-time delivery scoring, and transit risk analytics",
        "● Supplier Engine",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("Supply Chain Intelligence")
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
            st.dataframe(scored[available_cols], use_container_width=True, height=350)

            # Interactive Plotly visual analytics
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                fig_dist = create_supplier_risk_distribution(scored)
                st.plotly_chart(fig_dist, use_container_width=True)
            with col_sc2:
                fig_scatter = create_supplier_scatter(scored)
                st.plotly_chart(fig_scatter, use_container_width=True)
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
    render_page_header(
        "Inventory & Warehouse Optimization",
        "Stock health classification, days-of-supply tracking, and proactive stockout prevention",
        "● Inventory Engine",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("Inventory Optimization")
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
            fig_inv_donut = create_inventory_health_donut(status)
            st.plotly_chart(fig_inv_donut, use_container_width=True)
            st.dataframe(status, use_container_width=True, height=350)
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
            fig_stock_scatter = create_stockout_risk_scatter(stockout)
            st.plotly_chart(fig_stock_scatter, use_container_width=True)
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
    render_page_header(
        "Revenue & Financial Performance",
        "Sales trends, product margins, regional distribution, and revenue-at-risk breakdown",
        "● Revenue Engine",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("Revenue Analytics")
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
                fig_trend = create_revenue_trend_chart(trend)
                st.plotly_chart(fig_trend, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Revenue Overview: {e}")

    with tab2:
        try:
            products = get_revenue_by_product(
                data["orders"], data["order_items"], data["products"], days=90,
            )
            if not products.empty:
                fig_prod = create_product_revenue_bar(products, top_n=15)
                st.plotly_chart(fig_prod, use_container_width=True)
                st.dataframe(products, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Product Revenue: {e}")

    with tab3:
        try:
            regions = get_revenue_by_region(data["orders"], data["order_items"], days=90)
            if not regions.empty:
                fig_reg = create_regional_revenue_chart(regions)
                st.plotly_chart(fig_reg, use_container_width=True)
                st.dataframe(regions, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Regional Revenue: {e}")


def page_customers():
    """Customer analytics page."""
    render_page_header(
        "Customer Intelligence & Retention",
        "RFM behavioral segmentation, customer lifetime value (LTV) tiers, and churn probability modeling",
        "● Customer Engine",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("Customer Intelligence")
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
                st.markdown("### 🏆 Customer LTV Distribution")
                fig_ltv = create_customer_ltv_bar(ltv)
                st.plotly_chart(fig_ltv, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not compute Customer Summary: {e}")

    with tab2:
        try:
            segments = segment_customers(data["customers"], data["orders"], data["order_items"])
            if not segments.empty:
                st.markdown("### 🎯 RFM Behavioral Segmentation")
                fig_treemap = create_customer_segment_treemap(segments)
                st.plotly_chart(fig_treemap, use_container_width=True)
                st.dataframe(segments, use_container_width=True, height=350)
        except Exception as e:
            st.warning(f"⚠️ Could not segment Customers: {e}")

    with tab3:
        try:
            churn = calculate_churn_probability(data["customers"], data["orders"])
            if not churn.empty:
                risk_counts = churn["churn_risk"].value_counts()
                risk_df = pd.DataFrame({"risk_level": risk_counts.index, "count": risk_counts.values})
                from sciro.dashboard.charts import apply_theme
                import plotly.express as px
                fig_churn = px.pie(
                    risk_df,
                    names="risk_level",
                    values="count",
                    hole=0.6,
                    color="risk_level",
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#f43f5e"},
                )
                fig_churn = apply_theme(fig_churn, "Customer Churn Risk Breakdown", height=340)
                st.plotly_chart(fig_churn, use_container_width=True)

                high_churn = churn[churn["churn_risk"] == "High"]
                if not high_churn.empty:
                    st.markdown(f"### ⚠️ {len(high_churn)} High Churn Risk Customers")
                    st.dataframe(high_churn, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not calculate Churn Risk: {e}")


def page_ml_models():
    """ML models page (Module 3)."""
    render_page_header(
        "Machine Learning Predictive Center",
        "Train, evaluate, and forecast with demand, revenue, stockout, and supplier risk models",
        "● ML Models Ready",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("ML Predictive Center")
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
                    fig_imp = create_feature_importance_chart(imp)
                    st.plotly_chart(fig_imp, use_container_width=True)
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
                    fig_rev_fore = create_revenue_forecast_chart(forecast)
                    st.plotly_chart(fig_rev_fore, use_container_width=True)
                    st.markdown("### Revenue Forecast Data")
                    st.dataframe(forecast, use_container_width=True)
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
                    fig_imp3 = create_feature_importance_chart(imp)
                    st.plotly_chart(fig_imp3, use_container_width=True)
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
    render_page_header(
        "Simulation & Stress Testing Lab",
        "Monte Carlo supply chain disruption shocks, price elasticity tuning, and inventory reallocation",
        "● Simulation Engine",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("Simulation Lab")
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

                    st.markdown("### 📊 Distribution of Revenue Loss (Monte Carlo Simulation)")
                    fig_mc = create_monte_carlo_histogram(
                        result.total_lost_revenue_distribution,
                        p95_loss=result.revenue_at_risk_p95,
                        mean_loss=result.revenue_at_risk_mean,
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

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
    render_page_header(
        "AI Supply Chain Copilot",
        "Natural language conversational assistant for ad-hoc queries, risk inquiries, and instant insights",
        "● Copilot Active",
    )
    if not st.session_state.data_loaded:
        render_generic_empty_state("AI Copilot")
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
    render_page_header(
        "Settings & Governance",
        "User management, access controls, and runtime system configuration",
        "● Admin Portal",
    )

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

    # Top Brand Identity in Sidebar
    st.sidebar.markdown("""
    <div style='padding: 0.6rem 0 1rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 1rem;'>
        <div style='display: flex; align-items: center; gap: 11px;'>
            <div style='width: 36px; height: 36px; border-radius: 9px; background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.15rem; color: #ffffff; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);'>
                S
            </div>
            <div>
                <div style='font-size: 1.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; line-height: 1.1;'>
                    SCIRO <span style='color: #06b6d4;'>AI</span>
                </div>
                <div style='font-size: 0.68rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;'>
                    Supply Chain Intelligence
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prime Top Navigation
    pages = {
        ":material/dashboard: Executive Overview": page_executive_dashboard,
        ":material/cloud_upload: Data Ingestion": page_data_ingestion,
        ":material/hub: Supply Chain": page_supply_chain,
        ":material/inventory_2: Inventory Health": page_inventory,
        ":material/payments: Revenue Analytics": page_revenue,
        ":material/group: Customers": page_customers,
        ":material/psychology: ML Models": page_ml_models,
        ":material/tune: Simulation & Risk": page_simulation,
        ":material/smart_toy: AI Copilot": page_copilot,
        ":material/settings: Settings": page_settings,
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

    # Divider & Data Source Section
    st.sidebar.markdown("<div style='margin: 1.1rem 0; border-top: 1px solid rgba(255, 255, 255, 0.08);'></div>", unsafe_allow_html=True)
    data_sidebar()

    # Pinned User Profile & Logout at Bottom
    st.sidebar.markdown("<div style='margin: 1.2rem 0; border-top: 1px solid rgba(255, 255, 255, 0.08);'></div>", unsafe_allow_html=True)
    session = st.session_state.session_data
    role_color = "#38bdf8" if session['role'] == "admin" else ("#34d399" if session['role'] == "analyst" else "#fbbf24")
    st.sidebar.markdown(f"""
    <div style='background: #1e293b; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.65rem;'>
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div>
                <div style='color: #f8fafc; font-size: 0.85rem; font-weight: 600;'>{session['full_name']}</div>
                <div style='color: #94a3b8; font-size: 0.72rem; text-transform: capitalize;'>System {session['role']}</div>
            </div>
            <span class='badge' style='background: rgba(79, 70, 229, 0.15); color: {role_color}; border: 1px solid {role_color}40; font-size: 0.7rem;'>
                {session['role'].upper()}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button(":material/logout: Sign Out", use_container_width=True, key="btn_logout"):
        st.session_state.auth_manager.logout(st.session_state.auth_token)
        st.session_state.authenticated = False
        st.session_state.auth_token = None
        st.session_state.session_data = None
        st.rerun()

    # Render Active Page
    pages[selection]()


if __name__ == "__main__":
    main()
