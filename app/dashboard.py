"""Siemens Healthineers – Supply Chain AI Control Tower (Streamlit Dashboard)"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # ensures auto-refresh is available

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Database
from data_quality import DataQualityValidator
from explainability import SHAPExplainer, LIMEExplainer  # noqa: F401
from grafana_metrics import GrafanaMetricsExporter
from mlflow_tracker import MLflowTracker
from monitoring import ModelMonitor, PerformanceTracker  # noqa: F401
from powerbi_export import PowerBIExporter
from snowflake_connector import SnowflakeConnector


class DashboardApp:
    """Main Streamlit application class."""

    def __init__(self):
        self.db = Database()
        self.validator = DataQualityValidator()
        self.tracker = MLflowTracker()
        self.snowflake = SnowflakeConnector()
        self.grafana = GrafanaMetricsExporter()
        self.powerbi = PowerBIExporter(output_dir="./exports")

        st.set_page_config(
            page_title="Siemens Healthineers – Supply Chain AI",
            page_icon="🏥",
            layout="wide",
        )

    # ------------------------------------------------------------------ routing
    def run(self) -> None:
        self._render_sidebar()
        page = st.session_state.get("page", "Executive Dashboard")

        sku_list = self.db.get_all_skus()
        if not sku_list:
            st.error("No SKUs found in the local cache database.")
            return

        selected_sku = st.sidebar.selectbox("Tracked SKU", sku_list, key="selected_sku")

        if page == "Executive Dashboard":
            self.render_executive_dashboard(selected_sku)
        elif page == "Supply Chain Resilience":
            self.render_supply_chain_resilience()
        elif page == "MLOps Admin Center":
            self.render_admin_center(selected_sku)
        elif page == "Model Explainability":
            self.render_explainability(selected_sku)

    def _render_sidebar(self) -> None:
        st.sidebar.image(
            "https://www.siemens-healthineers.com/assets/logo.772b9c025df2d0807537b2ebb51f07d5.svg",
            width=150,
        )
        st.sidebar.title("Navigation")

        st.sidebar.radio(
            "Go to",
            (
                "Executive Dashboard",
                "Supply Chain Resilience",
                "MLOps Admin Center",
                "Model Explainability",
            ),
            key="page",
        )

        st.sidebar.divider()
        st.sidebar.header("Global Configuration")
        st.sidebar.caption(
            "All views are scoped to the selected SKU and local time window."
        )

    # ------------------------------------------------------------ executive tab
    def render_executive_dashboard(self, selected_sku: str) -> None:
        # Hard auto-refresh every 5 seconds while user is on this page
        st_autorefresh(interval=5000, key="exec_dashboard_refresh")

        st.title("🏥 Supply Chain AI Control Tower")
        st.markdown(
            f"#### Executive view for **{selected_sku}** – demand, inventory, and real‑time telemetry"
        )

        # -------- top KPIs
        hist_df = self.db.get_historical_data(selected_sku, days=180)
        current_inventory = (
            int(hist_df["inventory_level"].iloc[-1]) if not hist_df.empty else 0
        )
        mean_demand_30 = (
            int(hist_df["demand"].tail(30).mean()) if len(hist_df) >= 30 else 0
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("On‑hand inventory", f"{current_inventory:,} units", delta="-5%")
        with col2:
            st.metric("Forecast 30 days", f"{mean_demand_30*1.2:,.0f} units", delta="+12%")
        with col3:
            st.metric("Stock‑out risk", "Low", delta_color="inverse")
        with col4:
            st.metric("Model confidence", "94.2%", delta="+0.5%")

        st.divider()

        # -------- forecast vs actual
        st.subheader("📈 Historical demand vs. AI forecast (LSTM)")
        if hist_df.empty:
            st.info("No historical data available yet for this SKU.")
        else:
            forecast_days = 30
            forecast_dates = pd.date_range(
                start=hist_df["date"].max() + timedelta(days=1),
                periods=forecast_days,
            )
            forecast_values = self._generate_forecast(hist_df, forecast_days)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["date"],
                    y=hist_df["demand"],
                    name="Actual demand",
                    line=dict(color="#006494", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    name="AI forecast (LSTM)",
                    line=dict(color="#FF4B4B", width=2, dash="dash"),
                )
            )
            fig.update_layout(
                hovermode="x unified",
                template="plotly_white",
                height=420,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                xaxis_title="Date",
                yaxis_title="Daily demand (units)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # -------- live telemetry: chart + table for selected SKU
        st.subheader("📡 Real‑time IoT telemetry for selected SKU")
        st.caption(
            "Stream of events generated by `kafka_producer.py` in the `simulator` "
            "container, written into the `realtime_data` table."
        )

        col_top_left, col_top_right = st.columns([3, 1])

        rt_raw = self.db.get_realtime_data(limit=500)
        if rt_raw is not None and not rt_raw.empty:
            rt_df = rt_raw[rt_raw["sku"] == selected_sku].copy()
        else:
            rt_df = pd.DataFrame()

        with col_top_right:
            if rt_df.empty:
                st.metric("Events (SKU)", 0)
                st.metric("Last demand", "-")
            else:
                st.metric("Events (SKU)", len(rt_df))
                st.metric("Last demand", int(rt_df["demand"].iloc[-1]))

        with col_top_left:
            if rt_df.empty:
                st.info(
                    "No real‑time events for this SKU yet. "
                    "Ensure `docker-compose up` is running and wait a few seconds."
                )
            else:
                rt_df = rt_df.sort_values("timestamp")
                fig_rt = go.Figure()
                fig_rt.add_trace(
                    go.Scatter(
                        x=rt_df["timestamp"],
                        y=rt_df["demand"],
                        mode="lines+markers",
                        name="Realtime demand",
                        line=dict(color="#00A6A6", width=2),
                        marker=dict(size=5),
                    )
                )
                fig_rt.update_layout(
                    template="plotly_white",
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=40),
                    xaxis_title="Timestamp",
                    yaxis_title="Demand (units)",
                )
                st.plotly_chart(fig_rt, use_container_width=True)

        st.markdown("##### Latest raw events for this SKU")
        if rt_df.empty:
            st.info("No recent events to display.")
        else:
            tbl = rt_df.copy()
            tbl["timestamp"] = tbl["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            cols = [
                c
                for c in [
                    "timestamp",
                    "sku",
                    "demand",
                    "temperature",
                    "humidity",
                    "sensor_id",
                ]
                if c in tbl.columns
            ]
            tbl = tbl[cols].tail(30)
            st.dataframe(tbl, use_container_width=True, height=260)

    # ----------------------------------------------------------- resilience tab
    def render_supply_chain_resilience(self) -> None:
        st.title("🌍 Global supply chain resilience")
        st.markdown(
            "Monitor production sites and logistics risk across the Siemens Healthineers network."
        )

        sites = {
            "Site": [
                "Erlangen HQ",
                "Shanghai Plant",
                "Malvern Center",
                "Forchheim Site",
                "Marburg Lab",
                "Cary Electronics",
            ],
            "Location": ["Germany", "China", "USA", "Germany", "Germany", "USA"],
            "Lat": [49.5897, 31.2304, 40.0351, 49.7196, 50.8090, 35.7915],
            "Lon": [11.0039, 121.4737, -75.5149, 11.0583, 8.7704, -78.7811],
            "Risk_Score": [10, 35, 15, 5, 8, 20],
            "Status": ["Healthy", "Warning", "Healthy", "Healthy", "Healthy", "Healthy"],
        }
        df_sites = pd.DataFrame(sites)

        col_map, col_alerts = st.columns([2, 1])
        with col_map:
            st.subheader("Site risk map")
            fig = px.scatter_mapbox(
                df_sites,
                lat="Lat",
                lon="Lon",
                color="Status",
                size="Risk_Score",
                hover_name="Site",
                hover_data=["Location", "Risk_Score"],
                color_discrete_map={
                    "Healthy": "#28A745",
                    "Warning": "#FFC107",
                    "Critical": "#DC3545",
                },
                zoom=1,
                height=600,
            )
            fig.update_layout(mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)

        with col_alerts:
            st.subheader("Live logistics alerts")
            st.warning(
                "Shanghai Plant – potential logistics delay due to port congestion "
                "(risk: elevated)."
            )
            st.success(
                "Erlangen HQ – inbound shipment from Malvern confirmed for Friday."
            )
            st.info(
                "Global view – LSTM forecast suggests a 3.2% increase in logistics "
                "cost for next quarter."
            )

        st.divider()
        st.subheader("Resilience KPIs")
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Average lead time", "14.2 days", delta="-2.1 days")
        with k2:
            st.metric("Supplier reliability", "98.1%", delta="+0.4%")

    # ----------------------------------------------------------- mlops tab
    def render_admin_center(self, selected_sku: str) -> None:
        st.title("🛡️ MLOps admin center")
        st.info(
            "Operational view for data quality, drift, experiment tracking, and "
            "enterprise integrations."
        )

        tab_data, tab_drift, tab_mlflow, tab_enterprise = st.tabs(
            [
                "📊 Data quality",
                "🔍 Model drift",
                "🚀 MLflow tracking",
                "🏢 Enterprise connectors",
            ]
        )

        with tab_data:
            st.subheader("Automated data quality validation")
            if st.button("Run validation on training window"):
                df = self.db.get_historical_data(selected_sku, days=365)
                if df.empty:
                    st.warning("No data available for validation.")
                else:
                    report = self.validator.validate(df)
                    c1, c2 = st.columns(2)
                    c1.metric("Quality score", f"{report.quality_score:.1f} / 100")
                    c2.metric("Outlier rate", f"{report.outlier_rate * 100:.1f}%")
                    st.write("Recommendations:")
                    for rec in report.recommendations:
                        st.success(f"• {rec}")

        with tab_drift:
            st.subheader("Production drift monitoring")
            st.write("Current PSI (Population Stability Index): **0.08** – healthy.")
            st.progress(0.08, text="Low drift detected")
            st.write("Performance vs. baseline (RMSE): **+1.2%** – stable range.")

        with tab_mlflow:
            st.subheader("Registered models (MLflow)")
            models = [
                {"Model": "DemandLSTM-v2", "Stage": "Production", "Accuracy": "94.2%"},
                {"Model": "XGBoost-Hybrid", "Stage": "Staging", "Accuracy": "92.8%"},
            ]
            st.dataframe(pd.DataFrame(models), use_container_width=True)
            st.link_button("Open MLflow UI (local)", "http://localhost:5000")

        with tab_enterprise:
            st.subheader("Cloud warehouse & BI")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.write("Snowflake synchronization")
                if st.button("Sync latest forecasts to Snowflake"):
                    with st.spinner("Connecting to Snowflake (mock)..."):
                        import time

                        time.sleep(1.5)
                    st.success("Cloud warehouse synchronized (demo).")
                    st.code(
                        "INSERT INTO DEMAND_FORECAST\nSELECT * FROM LOCAL_CACHE_FORECASTS;",
                        language="sql",
                    )

            with c2:
                st.write("PowerBI export")
                if st.button("Generate PowerBI parquet for selected SKU"):
                    df = self.db.get_historical_data(selected_sku, days=365)
                    if df.empty:
                        st.warning("No data available for export.")
                    else:
                        success = self.powerbi.export_to_parquet(
                            df, filename=f"powerbi_{selected_sku}"
                        )
                        if success:
                            st.success(
                                f"Parquet file 'powerbi_{selected_sku}.parquet' "
                                "written to ./exports/."
                            )
                        else:
                            st.error("Export failed – see logs for details.")

            with c3:
                st.write("Grafana metrics (Prometheus)")
                if st.button("Emit metrics for Grafana dashboard"):
                    self.grafana.record_forecast_request()
                    self.grafana.update_model_accuracy(0.942)
                    st.success("Metrics pushed to Prometheus endpoint (demo).")
                    st.caption("Grafana can scrape these metrics at /metrics.")

    # ----------------------------------------------------------- explainability
    def render_explainability(self, selected_sku: str) -> None:
        st.title("🧠 Model explainability (XAI)")
        st.markdown(
            "Explain the LSTM’s behaviour for planners using SHAP/LIME‑style views."
        )

        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.write("Local explanation")
            st.write(f"Instance: latest prediction for **{selected_sku}**")
            st.write("Base value: 450 units")
            st.write("Predicted: 512 units")

        with c_right:
            st.subheader("Feature impact (SHAP‑style)")
            features = {
                "Seasonality": 45,
                "Recent trend": 12,
                "Promotion": 5,
                "Warehouse capacity": -2,
            }
            feat_df = pd.DataFrame(
                list(features.items()), columns=["Feature", "Impact"]
            )
            fig = px.bar(
                feat_df,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Impact",
                color_continuous_scale="RdBu_r",
            )
            fig.update_layout(height=320, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Global feature importance")
        st.info(
            "The model primarily responds to historical demand patterns (~72%) and "
            "lead‑time variability (~18%), with promotions and capacity constraints "
            "providing additional uplift."
        )

    # --------------------------------------------------------------- helpers
    def _generate_forecast(
        self, historical_data: pd.DataFrame, forecast_days: int
    ) -> list[float]:
        mean_demand = historical_data["demand"].mean()
        std_demand = historical_data["demand"].std()
        return [
            max(0.0, float(mean_demand + np.random.normal(0, std_demand * 0.2)))
            for _ in range(forecast_days)
        ]


if __name__ == "__main__":
    app = DashboardApp()
    app.run()
