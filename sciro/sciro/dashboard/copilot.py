"""
SCIRO AI Copilot.

Natural language query assistant that interprets user questions
and routes them to the appropriate analytics, ML, or simulation engine.

This is a rule-based NLU system (no external LLM API required).
It uses keyword matching and intent classification to understand
user queries and return structured responses.

If an OPENAI_API_KEY is configured in .env, it can optionally
use GPT for more natural responses.
"""

import logging
import re
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Intent definitions
# =============================================================================
INTENTS = {
    "revenue_summary": {
        "keywords": ["revenue", "sales", "income", "earnings", "turnover", "total revenue"],
        "description": "Get revenue KPIs and summary",
    },
    "revenue_trend": {
        "keywords": ["revenue trend", "monthly revenue", "monthly sales", "revenue growth",
                      "sales trend", "revenue over time"],
        "description": "Revenue trend over time",
    },
    "top_products": {
        "keywords": ["top product", "best selling", "popular product", "product revenue",
                      "product performance", "best product", "top selling products",
                      "selling products", "top products"],
        "description": "Top performing products",
    },
    "supplier_risk": {
        "keywords": ["supplier risk", "risky supplier", "supplier danger", "supplier problem",
                      "supplier performance", "supplier reliable", "supplier issue",
                      "risky", "supplier", "suppliers"],
        "description": "Supplier risk analysis",
    },
    "revenue_at_risk": {
        "keywords": ["revenue at risk", "revenue risk", "risk revenue", "money at risk",
                      "what if supplier", "supplier impact", "supplier fail"],
        "description": "Revenue-at-risk analysis",
    },
    "inventory_status": {
        "keywords": ["inventory", "stock", "stockout", "out of stock", "low stock",
                      "stock level", "warehouse", "inventory health"],
        "description": "Inventory status and health",
    },
    "customer_analysis": {
        "keywords": ["customer", "client", "buyer", "who buys", "customer segment",
                      "churn", "customer value", "ltv", "lifetime value"],
        "description": "Customer analytics",
    },
    "demand_forecast": {
        "keywords": ["forecast", "predict demand", "future demand", "demand prediction",
                      "what will sell", "expected demand"],
        "description": "Demand forecasting",
    },
    "pricing": {
        "keywords": ["price", "pricing", "optimal price", "price change",
                      "price optimization", "margin", "markup"],
        "description": "Pricing optimization",
    },
    "simulation": {
        "keywords": ["simulate", "simulation", "what if", "scenario", "disruption",
                      "monte carlo", "worst case"],
        "description": "Disruption simulation",
    },
    "data_upload": {
        "keywords": ["upload", "import", "excel", "pdf", "sheet", "uploading", "file upload", "upload excel", "upload pdf"],
        "description": "Upload and import Excel or PDF data",
    },
    "help": {
        "keywords": ["help", "what can you do", "commands", "how to", "guide"],
        "description": "Show available commands",
    },
}


class AICopilot:
    """Natural language query assistant."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._handlers: Dict[str, Callable] = {
            "revenue_summary": self._handle_revenue_summary,
            "revenue_trend": self._handle_revenue_trend,
            "top_products": self._handle_top_products,
            "supplier_risk": self._handle_supplier_risk,
            "revenue_at_risk": self._handle_revenue_at_risk,
            "inventory_status": self._handle_inventory_status,
            "customer_analysis": self._handle_customer_analysis,
            "demand_forecast": self._handle_demand_forecast,
            "pricing": self._handle_pricing,
            "simulation": self._handle_simulation,
            "data_upload": self._handle_data_upload,
            "help": self._handle_help,
        }

    def load_data(self, data_dir: str) -> None:
        """Load all CSV data for analytics."""
        import os
        self.data_dir = data_dir
        for name in ("regions", "suppliers", "products", "inventory",
                      "shipments", "customers", "orders", "order_items"):
            path = os.path.join(data_dir, f"{name}.csv")
            if os.path.exists(path):
                self._data_cache[name] = pd.read_csv(path)
                # Parse dates
                if name == "orders" and "order_date" in self._data_cache[name].columns:
                    self._data_cache[name]["order_date"] = pd.to_datetime(
                        self._data_cache[name]["order_date"], errors="coerce"
                    )
                logger.info("Copilot loaded %s: %d rows", name, len(self._data_cache[name]))

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a natural language query and return a response.

        Args:
            user_query: Natural language question from the user.

        Returns:
            Dict with:
            - intent: Detected intent
            - response: Human-readable text response
            - data: Optional structured data (dict/DataFrame)
            - visualization_hint: Suggested chart type
        """
        intent, confidence = self._classify_intent(user_query)

        response = {
            "query": user_query,
            "intent": intent,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "response": "",
            "data": None,
            "visualization_hint": None,
        }

        if confidence < 0.3:
            response["response"] = (
                "I'm not sure I understand your question. Here's what I can help with:\n"
                + self._get_help_text()
            )
            response["intent"] = "unclear"
            return response

        # Check if data is loaded
        if intent != "help" and not self._data_cache:
            response["response"] = (
                "⚠️ No data loaded. Please load data first using the 'Load Data' button "
                "in the dashboard, or call copilot.load_data(data_dir)."
            )
            return response

        # Route to handler
        handler = self._handlers.get(intent, self._handle_help)
        try:
            result = handler(user_query)
            response.update(result)
        except Exception as e:
            logger.error("Copilot error handling '%s': %s", intent, e)
            response["response"] = f"Sorry, I encountered an error: {str(e)}"

        return response

    def _classify_intent(self, query: str) -> tuple:
        """Classify user query into an intent with confidence score."""
        query_lower = query.lower().strip()
        best_intent = "help"
        best_score = 0

        for intent, config in INTENTS.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    # Weight by keyword length (longer matches are better)
                    score += len(keyword.split())

            if score > best_score:
                best_score = score
                best_intent = intent

        # Normalize confidence (rough heuristic)
        confidence = min(1.0, best_score / 3.0) if best_score > 0 else 0.0
        return best_intent, round(confidence, 2)

    def _extract_number(self, query: str, default: int = 30) -> int:
        """Extract a number from the query (e.g. 'last 60 days')."""
        match = re.search(r"(\d+)\s*(day|month|week)", query.lower())
        if match:
            return int(match.group(1))
        return default

    def _extract_supplier_name(self, query: str) -> Optional[str]:
        """Try to extract a supplier name from the query."""
        if "suppliers" not in self._data_cache:
            return None
        suppliers = self._data_cache["suppliers"]
        for _, row in suppliers.iterrows():
            if row["name"].lower() in query.lower():
                return row["name"]
        return None

    # =============================================================================
    # Intent Handlers
    # =============================================================================
    def _handle_revenue_summary(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.revenue import get_revenue_summary
        days = self._extract_number(query, 30)
        summary = get_revenue_summary(
            self._data_cache["orders"], self._data_cache["order_items"],
            self._data_cache["products"], days=days,
        )
        response = (
            f"📊 **Revenue Summary (Last {days} Days)**\n\n"
            f"• Total Revenue: ₹{summary['total_revenue']:,.2f}\n"
            f"• Total Orders: {summary['total_orders']}\n"
            f"• Average Order Value: ₹{summary['avg_order_value']:,.2f}\n"
            f"• Units Sold: {summary['total_units_sold']}\n"
            f"• Gross Profit: ₹{summary['gross_profit']:,.2f}\n"
            f"• Gross Margin: {summary['gross_margin_pct']:.1f}%"
        )
        return {"response": response, "data": summary, "visualization_hint": "kpi_cards"}

    def _handle_revenue_trend(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.revenue import get_monthly_revenue_trend
        months = min(self._extract_number(query, 12), 24)
        trend = get_monthly_revenue_trend(
            self._data_cache["orders"], self._data_cache["order_items"], months=months,
        )
        if trend.empty:
            return {"response": "No revenue trend data available for the specified period."}

        response = (
            f"📈 **Revenue Trend (Last {months} Months)**\n\n"
            f"• Latest Month Revenue: ₹{trend.iloc[-1]['revenue']:,.2f}\n"
            f"• Overall period: {trend.iloc[0]['month']} to {trend.iloc[-1]['month']}\n"
            f"• Avg Monthly Revenue: ₹{trend['revenue'].mean():,.2f}\n"
            f"• Peak: ₹{trend['revenue'].max():,.2f}"
        )
        return {
            "response": response,
            "data": trend.to_dict("records"),
            "visualization_hint": "line_chart",
        }

    def _handle_top_products(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.revenue import get_revenue_by_product
        top_n = self._extract_number(query, 10)
        products = get_revenue_by_product(
            self._data_cache["orders"], self._data_cache["order_items"],
            self._data_cache["products"], days=90, top_n=top_n,
        )
        if products.empty:
            return {"response": "No product data available."}

        lines = [f"🏆 **Top {min(top_n, len(products))} Products by Revenue**\n"]
        for _, row in products.head(top_n).iterrows():
            lines.append(
                f"  {int(row['rank'])}. **{row['product_name']}** — "
                f"₹{row['revenue']:,.0f} ({row['units_sold']} units, {row['margin_pct']:.1f}% margin)"
            )
        return {
            "response": "\n".join(lines),
            "data": products.to_dict("records"),
            "visualization_hint": "bar_chart",
        }

    def _handle_supplier_risk(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.supply_chain import get_supply_chain_summary
        summary = get_supply_chain_summary(
            self._data_cache["suppliers"], self._data_cache["shipments"],
            self._data_cache["products"],
        )
        response = (
            f"⚠️ **Supplier Risk Overview**\n\n"
            f"• Total Suppliers: {summary['total_suppliers']} "
            f"({summary['active_suppliers']} active)\n"
            f"• High Risk: {summary['high_risk_supplier_count']}\n"
            f"• Critical Risk: {summary['critical_risk_supplier_count']}\n"
            f"• Avg On-Time Delivery: {summary['avg_on_time_delivery_pct']}%\n"
        )
        if summary["top_risk_suppliers"]:
            response += "\n**Top Risk Suppliers:**\n"
            for s in summary["top_risk_suppliers"][:5]:
                response += (
                    f"  • {s['supplier_name']} — Score: {s['risk_score']}/100 "
                    f"({s['risk_level']})\n"
                )
        return {"response": response, "data": summary, "visualization_hint": "bar_chart"}

    def _handle_revenue_at_risk(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.risk_engine import get_risk_dashboard_data
        dashboard = get_risk_dashboard_data(
            self._data_cache["suppliers"], self._data_cache["products"],
            self._data_cache["inventory"], self._data_cache["orders"],
            self._data_cache["order_items"], self._data_cache["shipments"],
        )
        response = (
            f"💰 **Revenue-at-Risk Analysis**\n\n"
            f"• Total Revenue at Risk: ₹{dashboard['total_revenue_at_risk']:,.2f}\n"
            f"• High-Risk Suppliers: {dashboard['high_risk_supplier_count']}\n"
            f"• Products at Risk: {dashboard['total_products_at_risk']}\n"
            f"• Affected Customers: {dashboard['total_affected_customers']}\n"
        )
        return {"response": response, "data": dashboard, "visualization_hint": "risk_gauge"}

    def _handle_inventory_status(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.inventory import get_inventory_summary
        summary = get_inventory_summary(
            self._data_cache["inventory"], self._data_cache["products"],
            self._data_cache["orders"], self._data_cache["order_items"],
            self._data_cache.get("suppliers"),
        )
        response = (
            f"📦 **Inventory Status**\n\n"
            f"• Total Records: {summary['total_inventory_records']}\n"
            f"• Total Value: ₹{summary['total_inventory_value']:,.2f}\n"
            f"• Healthy: {summary['healthy_stock_count']}\n"
            f"• Low Stock: {summary['low_stock_count']}\n"
            f"• Critical: {summary['critical_stock_count']}\n"
            f"• Out of Stock: {summary['out_of_stock_count']}\n"
            f"• High Stockout Risk: {summary['high_stockout_risk_count']}\n"
            f"• Potential Lost Revenue: ₹{summary['total_potential_lost_revenue']:,.2f}\n"
        )
        return {"response": response, "data": summary, "visualization_hint": "pie_chart"}

    def _handle_customer_analysis(self, query: str) -> Dict[str, Any]:
        from sciro.analytics.customer import get_customer_summary
        summary = get_customer_summary(
            self._data_cache["customers"], self._data_cache["orders"],
            self._data_cache["order_items"],
        )
        response = (
            f"👥 **Customer Analysis**\n\n"
            f"• Total Customers: {summary['total_customers']}\n"
            f"• Active (last 90 days): {summary['active_customers']}\n"
            f"• New (last 30 days): {summary['new_customers']}\n"
            f"• At Risk (high churn): {summary['at_risk_customers']}\n"
            f"• Avg Customer LTV: ₹{summary['avg_customer_ltv']:,.2f}\n"
        )
        if summary.get("customers_by_segment"):
            response += "\n**Segments:**\n"
            for seg, count in summary["customers_by_segment"].items():
                response += f"  • {seg}: {count}\n"
        return {"response": response, "data": summary, "visualization_hint": "pie_chart"}

    def _handle_demand_forecast(self, query: str) -> Dict[str, Any]:
        response = (
            "🔮 **Demand Forecasting**\n\n"
            "To run demand forecasting:\n"
            "1. Go to the **ML Models** tab in the dashboard\n"
            "2. Select **Demand Forecaster**\n"
            "3. Click **Train & Predict**\n\n"
            "The model uses gradient boosting with time-series features "
            "(lags, rolling averages, calendar features) to predict future demand per product."
        )
        return {"response": response, "visualization_hint": "line_chart"}

    def _handle_pricing(self, query: str) -> Dict[str, Any]:
        response = (
            "💲 **Pricing Optimization**\n\n"
            "To optimize pricing:\n"
            "1. Go to the **Optimization** tab in the dashboard\n"
            "2. Select **Pricing Optimizer**\n"
            "3. Choose objective (Revenue or Profit)\n"
            "4. Click **Run Optimization**\n\n"
            "The optimizer estimates price elasticity from historical data "
            "and finds optimal prices across the product range."
        )
        return {"response": response, "visualization_hint": None}

    def _handle_simulation(self, query: str) -> Dict[str, Any]:
        response = (
            "🎲 **Disruption Simulation**\n\n"
            "To run Monte Carlo simulations:\n"
            "1. Go to the **Simulation** tab in the dashboard\n"
            "2. Configure scenario (supplier failure, demand surge, etc.)\n"
            "3. Set number of simulations and horizon\n"
            "4. Click **Run Simulation**\n\n"
            "The simulator models thousands of scenarios to estimate "
            "revenue at risk and identify vulnerable products."
        )
        return {"response": response, "visualization_hint": "histogram"}

    def _handle_data_upload(self, query: str) -> Dict[str, Any]:
        response = (
            "📤 **Uploading Data to SCIRO**\n\n"
            "You can upload data in two ways:\n"
            "1. **Quick Sidebar Upload**: Open the 'Quick File Upload' section in the left sidebar to upload any Excel (.xlsx, .xls), PDF (.pdf), or CSV file.\n"
            "2. **Full Ingestion Hub**: Navigate to the **📁 Data Upload & Ingestion** page in the main navigation. You can:\n"
            "   • Select specific Excel sheets or extract PDF tables\n"
            "   • Map data to standard entities (`suppliers`, `products`, `orders`, `inventory`, etc.)\n"
            "   • Run automatic Pydantic schema validation and ETL quality checks\n"
            "   • Load the dataset directly into active memory or save to disk\n\n"
            "All dashboards, predictions, and simulations immediately use your uploaded data!"
        )
        return {"response": response, "visualization_hint": None}

    def _handle_help(self, query: str) -> Dict[str, Any]:
        return {"response": self._get_help_text(), "visualization_hint": None}

    def _get_help_text(self) -> str:
        lines = [
            "🤖 **SCIRO AI Copilot — Available Queries**\n",
            "Try asking questions like:\n",
            '• "What\'s our revenue for the last 30 days?"',
            '• "Show me the monthly revenue trend"',
            '• "What are the top 10 selling products?"',
            '• "Which suppliers are high risk?"',
            '• "How much revenue is at risk?"',
            '• "What\'s the inventory status?"',
            '• "Show me customer analysis"',
            '• "How do I run demand forecasting?"',
            '• "How do I optimize pricing?"',
            '• "How do I run a disruption simulation?"',
        ]
        return "\n".join(lines)
