"""
SCIRO Dashboard Charting & Visualization Library.

Provides enterprise-grade, colorful, interactive Plotly visualizations
with a unified dark design system, glowing accents, and formatted currency tooltips.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Color Palettes & Design Tokens
# =============================================================================
COLOR_PALETTE = [
    "#06b6d4",  # Cyan
    "#6366f1",  # Indigo
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#ec4899",  # Pink
    "#8b5cf6",  # Violet
    "#3b82f6",  # Blue
    "#14b8a6",  # Teal
]

RISK_COLORS = {
    "LOW": "#10b981",
    "MEDIUM": "#f59e0b",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
    "Low": "#10b981",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
    "Healthy": "#10b981",
    "HEALTHY": "#10b981",
    "Reorder Required": "#f59e0b",
    "REORDER_REQUIRED": "#f59e0b",
    "Critical Shortage": "#f97316",
    "CRITICAL_SHORTAGE": "#f97316",
    "Out of Stock": "#ef4444",
    "OUT_OF_STOCK": "#ef4444",
}

SEGMENT_COLORS = {
    "Champions": "#10b981",
    "Loyal Customers": "#06b6d4",
    "Potential Loyalists": "#3b82f6",
    "Recent Customers": "#8b5cf6",
    "Promising": "#a855f7",
    "Customers Needing Attention": "#f59e0b",
    "About To Sleep": "#fb923c",
    "At Risk": "#f97316",
    "Can't Lose Them": "#e11d48",
    "Hibernating": "#94a3b8",
    "Lost": "#64748b",
}


def apply_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply consistent high-contrast dark theme styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": "#f8fafc", "family": "Inter, sans-serif"},
        } if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, -apple-system, sans-serif", "color": "#94a3b8", "size": 12},
        height=height,
        margin={"l": 40, "r": 30, "t": 50 if title else 25, "b": 40},
        hoverlabel={
            "bgcolor": "#0f172a",
            "bordercolor": "#38bdf8",
            "font": {"color": "#f8fafc", "size": 12, "family": "Inter, sans-serif"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": "#cbd5e1"},
        },
    )
    fig.update_xaxes(
        gridcolor="#1e293b",
        zerolinecolor="#334155",
        tickfont={"color": "#94a3b8"},
        title_font={"color": "#cbd5e1", "size": 12},
    )
    fig.update_yaxes(
        gridcolor="#1e293b",
        zerolinecolor="#334155",
        tickfont={"color": "#94a3b8"},
        title_font={"color": "#cbd5e1", "size": 12},
    )
    return fig


# =============================================================================
# Executive & Revenue Charts
# =============================================================================
def create_revenue_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Create a smooth, colorful area chart for monthly revenue trends."""
    if trend_df.empty or "month" not in trend_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Revenue Trend (No data)")

    df = trend_df.copy()
    df["month_str"] = df["month"].astype(str)

    fig = go.Figure()

    # Revenue Area with Gradient fill
    fig.add_trace(
        go.Scatter(
            x=df["month_str"],
            y=df["revenue"],
            mode="lines+markers",
            name="Revenue (₹)",
            line={"color": "#06b6d4", "width": 3, "shape": "spline"},
            marker={"size": 7, "color": "#38bdf8", "line": {"width": 2, "color": "#ffffff"}},
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.15)",
            hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
        )
    )

    # Secondary trace for Orders if available
    if "orders" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["month_str"],
                y=df["orders"],
                mode="lines+markers",
                name="Order Count",
                yaxis="y2",
                line={"color": "#f59e0b", "width": 2, "dash": "dot"},
                marker={"size": 6, "color": "#fbbf24"},
                hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis2={
                "title": "Orders",
                "overlaying": "y",
                "side": "right",
                "gridcolor": "rgba(0,0,0,0)",
                "tickfont": {"color": "#fbbf24"},
                "title_font": {"color": "#fbbf24"},
            }
        )

    fig.update_layout(
        yaxis={"title": "Revenue (₹)", "tickprefix": "₹"},
        xaxis={"title": "Month"},
    )
    return apply_theme(fig, "Monthly Revenue Trend & Volume", height=380)


def create_revenue_at_risk_donut(risk_data: Dict[str, Any]) -> go.Figure:
    """Create an interactive donut chart of revenue at risk breakdown."""
    supplier_risks = risk_data.get("supplier_risks", [])
    if not supplier_risks:
        fig = go.Figure()
        return apply_theme(fig, "Revenue at Risk (No data)")

    df = pd.DataFrame(supplier_risks)
    if "risk_level" not in df.columns or "revenue_at_risk" not in df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Revenue at Risk Breakdown")

    grouped = df.groupby("risk_level")["revenue_at_risk"].sum().reset_index()
    colors = [RISK_COLORS.get(level, "#6366f1") for level in grouped["risk_level"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=grouped["risk_level"],
                values=grouped["revenue_at_risk"],
                hole=0.65,
                marker={"colors": colors, "line": {"color": "#0f172a", "width": 2}},
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Revenue at Risk: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    total_risk = risk_data.get("total_revenue_at_risk", grouped["revenue_at_risk"].sum())
    fig.add_annotation(
        text=f"₹{total_risk/100000:,.1f}L<br><span style='font-size:11px;color:#94a3b8;'>Total At Risk</span>",
        x=0.5,
        y=0.5,
        font={"size": 18, "color": "#f43f5e", "family": "Inter, sans-serif"},
        showarrow=False,
    )
    return apply_theme(fig, "Revenue at Risk by Supplier Threat Tier", height=340)


def create_product_revenue_bar(products_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Create a vibrant horizontal bar chart of top performing products."""
    if products_df.empty:
        fig = go.Figure()
        return apply_theme(fig, "Top Products by Revenue (No data)")

    df = products_df.head(top_n).copy()
    name_col = "name" if "name" in df.columns else ("product_name" if "product_name" in df.columns else "product_sku")
    rev_col = "revenue" if "revenue" in df.columns else "total_revenue"

    if rev_col not in df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Top Products")

    df = df.sort_values(rev_col, ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df[rev_col],
            y=df[name_col],
            orientation="h",
            marker={
                "color": df[rev_col],
                "colorscale": [[0, "#38bdf8"], [0.5, "#6366f1"], [1, "#ec4899"]],
                "line": {"color": "#ffffff", "width": 0.5},
            },
            hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Gross Revenue (₹)", "tickprefix": "₹"},
        yaxis={"title": ""},
    )
    return apply_theme(fig, f"Top {top_n} Revenue Generating Products", height=420)


def create_regional_revenue_chart(regions_df: pd.DataFrame) -> go.Figure:
    """Create a colorful donut or bar breakdown of regional sales performance."""
    if regions_df.empty:
        fig = go.Figure()
        return apply_theme(fig, "Regional Revenue (No data)")

    region_col = "region_name" if "region_name" in regions_df.columns else "region"
    rev_col = "revenue" if "revenue" in regions_df.columns else "total_revenue"

    fig = px.bar(
        regions_df,
        x=region_col,
        y=rev_col,
        color=region_col,
        color_discrete_sequence=COLOR_PALETTE,
        text_auto=".2s",
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
        marker={"line": {"color": "#1e293b", "width": 1}},
    )
    fig.update_layout(
        showlegend=False,
        xaxis={"title": "Region"},
        yaxis={"title": "Revenue (₹)", "tickprefix": "₹"},
    )
    return apply_theme(fig, "Revenue Contribution by Geographic Region", height=380)


# =============================================================================
# Supply Chain Visuals
# =============================================================================
def create_supplier_risk_distribution(scored_df: pd.DataFrame) -> go.Figure:
    """Create a colored bar chart showing supplier counts across risk tiers."""
    if scored_df.empty or "risk_level" not in scored_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Supplier Risk Distribution")

    counts = scored_df["risk_level"].value_counts().reset_index()
    counts.columns = ["risk_level", "count"]
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts["sort_order"] = counts["risk_level"].apply(lambda x: order.index(x) if x in order else 99)
    counts = counts.sort_values("sort_order")

    colors = [RISK_COLORS.get(level, "#6366f1") for level in counts["risk_level"]]

    fig = go.Figure(
        go.Bar(
            x=counts["risk_level"],
            y=counts["count"],
            marker={"color": colors, "line": {"color": "#ffffff", "width": 1}},
            text=counts["count"],
            textposition="auto",
            hovertemplate="<b>%{x} Risk</b><br>Suppliers: %{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis={"title": "Supplier Risk Level"}, yaxis={"title": "Supplier Count"})
    return apply_theme(fig, "Supplier Portfolio Risk Profile", height=350)


def create_supplier_scatter(scored_df: pd.DataFrame) -> go.Figure:
    """Create a multi-dimensional scatter matrix of suppliers."""
    if scored_df.empty:
        fig = go.Figure()
        return apply_theme(fig, "Supplier Reliability Matrix")

    req_cols = ["on_time_delivery_pct", "avg_delay_days", "supplier_name", "risk_level"]
    if not all(c in scored_df.columns for c in req_cols):
        fig = go.Figure()
        return apply_theme(fig, "Supplier Reliability Matrix")

    fig = px.scatter(
        scored_df,
        x="avg_delay_days",
        y="on_time_delivery_pct",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        size=scored_df["risk_score"].clip(lower=10) if "risk_score" in scored_df.columns else None,
        hover_name="supplier_name",
        hover_data={"on_time_delivery_pct": ":.1f%", "avg_delay_days": ":.1f", "risk_level": True},
    )
    fig.update_traces(marker={"line": {"width": 1, "color": "#ffffff"}, "opacity": 0.85})
    fig.update_layout(
        xaxis={"title": "Average Delivery Delay (Days)"},
        yaxis={"title": "On-Time Delivery Rate (%)", "ticksuffix": "%"},
    )
    return apply_theme(fig, "Supplier Delivery Performance vs Delays", height=400)


# =============================================================================
# Inventory Visuals
# =============================================================================
def create_inventory_health_donut(status_df: pd.DataFrame) -> go.Figure:
    """Create a donut chart of inventory stock health classification."""
    if status_df.empty or "stock_status" not in status_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Inventory Health (No data)")

    counts = status_df["stock_status"].value_counts().reset_index()
    counts.columns = ["status", "count"]
    colors = [RISK_COLORS.get(s, "#06b6d4") for s in counts["status"]]

    fig = go.Figure(
        go.Pie(
            labels=counts["status"],
            values=counts["count"],
            hole=0.6,
            marker={"colors": colors, "line": {"color": "#0f172a", "width": 2}},
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>SKUs: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
    )
    total_skus = len(status_df)
    fig.add_annotation(
        text=f"{total_skus:,}<br><span style='font-size:11px;color:#94a3b8;'>Total SKUs</span>",
        x=0.5,
        y=0.5,
        font={"size": 18, "color": "#38bdf8", "family": "Inter, sans-serif"},
        showarrow=False,
    )
    return apply_theme(fig, "Inventory SKU Stock Health Classification", height=360)


def create_stockout_risk_scatter(stockout_df: pd.DataFrame) -> go.Figure:
    """Create an interactive scatter plot of stockout probability vs days of supply."""
    if stockout_df.empty or "days_of_supply" not in stockout_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Stockout Risk vs Days of Supply")

    df = stockout_df.copy()
    df["stockout_prob_pct"] = (df["stockout_probability"] * 100).round(1)

    fig = px.scatter(
        df,
        x="days_of_supply",
        y="stockout_prob_pct",
        color="stockout_risk" if "stockout_risk" in df.columns else None,
        color_discrete_map=RISK_COLORS,
        hover_name="name" if "name" in df.columns else "product_sku",
        hover_data={"days_of_supply": ":.1f", "stockout_prob_pct": ":.1f%"},
    )
    fig.update_traces(marker={"size": 9, "line": {"width": 1, "color": "#ffffff"}})
    fig.update_layout(
        xaxis={"title": "Days of Supply (Inventory Buffer)"},
        yaxis={"title": "Stockout Likelihood (%)", "ticksuffix": "%"},
    )
    return apply_theme(fig, "Stockout Risk Probability vs Days of Supply", height=380)


# =============================================================================
# Customer Visuals
# =============================================================================
def create_customer_segment_treemap(segments_df: pd.DataFrame) -> go.Figure:
    """Create a colorful treemap of RFM customer segments."""
    if segments_df.empty or "segment" not in segments_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Customer RFM Segmentation")

    grouped = segments_df.groupby("segment").agg(
        customer_count=("customer_id" if "customer_id" in segments_df.columns else segments_df.columns[0], "count"),
        avg_spend=("monetary" if "monetary" in segments_df.columns else "total_spend", "mean") if "monetary" in segments_df.columns or "total_spend" in segments_df.columns else ("customer_id", "count"),
    ).reset_index()

    colors = [SEGMENT_COLORS.get(s, "#6366f1") for s in grouped["segment"]]

    fig = px.treemap(
        grouped,
        path=["segment"],
        values="customer_count",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percentRoot:.1%}<extra></extra>",
        marker={"line": {"width": 2, "color": "#0f172a"}},
    )
    return apply_theme(fig, "Customer RFM Behavioral Segments", height=380)


def create_customer_ltv_bar(ltv_df: pd.DataFrame) -> go.Figure:
    """Create a gradient bar chart of customer lifetime value tiers."""
    if ltv_df.empty or "clv_tier" not in ltv_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Customer LTV Tiers")

    counts = ltv_df["clv_tier"].value_counts().reset_index()
    counts.columns = ["tier", "count"]
    tier_order = ["Platinum", "Gold", "Silver", "Bronze"]
    counts["sort_order"] = counts["tier"].apply(lambda x: tier_order.index(x) if x in tier_order else 99)
    counts = counts.sort_values("sort_order")

    palette = {"Platinum": "#a855f7", "Gold": "#f59e0b", "Silver": "#38bdf8", "Bronze": "#fb923c"}
    colors = [palette.get(t, "#6366f1") for t in counts["tier"]]

    fig = go.Figure(
        go.Bar(
            x=counts["tier"],
            y=counts["count"],
            marker={"color": colors, "line": {"color": "#ffffff", "width": 1}},
            text=counts["count"],
            textposition="auto",
            hovertemplate="<b>%{x} Tier</b><br>Customers: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(xaxis={"title": "CLV Tier"}, yaxis={"title": "Customer Count"})
    return apply_theme(fig, "Customer Lifetime Value (CLV) Tier Distribution", height=340)


# =============================================================================
# Machine Learning & Forecast Visuals
# =============================================================================
def create_feature_importance_chart(imp_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create a sleek horizontal bar chart of ML feature importances."""
    if imp_df.empty or "feature" not in imp_df.columns or "importance" not in imp_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Feature Importance")

    df = imp_df.head(top_n).sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker={
                "color": df["importance"],
                "colorscale": [[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#06b6d4"]],
                "line": {"color": "#ffffff", "width": 0.5},
            },
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis={"title": "Relative Feature Importance Score"}, yaxis={"title": ""})
    return apply_theme(fig, f"Top {top_n} Predictive Model Drivers", height=380)


def create_revenue_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Create a revenue prediction curve."""
    if forecast_df.empty or "predicted_revenue" not in forecast_df.columns:
        fig = go.Figure()
        return apply_theme(fig, "Revenue Forecast")

    df = forecast_df.copy()
    x_col = "month" if "month" in df.columns else df.columns[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col].astype(str),
            y=df["predicted_revenue"],
            mode="lines+markers",
            name="Forecasted Revenue",
            line={"color": "#10b981", "width": 3, "dash": "dash"},
            marker={"size": 8, "color": "#34d399", "line": {"width": 2, "color": "#ffffff"}},
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)",
            hovertemplate="<b>%{x}</b><br>Projected: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Forecast Horizon (Months)"},
        yaxis={"title": "Projected Revenue (₹)", "tickprefix": "₹"},
    )
    return apply_theme(fig, "AI Machine Learning Revenue Projections", height=380)


# =============================================================================
# Simulation Visuals
# =============================================================================
def create_monte_carlo_histogram(losses: List[float], p95_loss: float = 0.0, mean_loss: float = 0.0) -> go.Figure:
    """Create an interactive probability density histogram for Monte Carlo disruptions."""
    if not losses:
        fig = go.Figure()
        return apply_theme(fig, "Disruption Simulation Distribution")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=losses,
            nbinsx=40,
            marker={
                "color": "rgba(244, 63, 94, 0.75)",
                "line": {"color": "#f43f5e", "width": 1},
            },
            name="Simulated Trials",
            hovertemplate="Revenue Loss: ₹%{x:,.0f}<br>Trial Frequency: %{y}<extra></extra>",
        )
    )

    if mean_loss > 0:
        fig.add_vline(
            x=mean_loss,
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=f"Mean: ₹{mean_loss:,.0f}",
            annotation_position="top left",
            annotation_font_color="#38bdf8",
        )
    if p95_loss > 0:
        fig.add_vline(
            x=p95_loss,
            line_dash="dot",
            line_color="#fb7185",
            annotation_text=f"P95 VaR: ₹{p95_loss:,.0f}",
            annotation_position="top right",
            annotation_font_color="#fb7185",
        )

    fig.update_layout(
        xaxis={"title": "Simulated Revenue Loss Exposure (₹)", "tickprefix": "₹"},
        yaxis={"title": "Frequency (Simulation Trials)"},
    )
    return apply_theme(fig, "Monte Carlo Disruption Revenue Loss Distribution", height=380)
