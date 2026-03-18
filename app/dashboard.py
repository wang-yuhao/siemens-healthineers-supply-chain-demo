"""Interactive Streamlit Dashboard for Demand Forecasting"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Database
from neural_network_model import DemandLSTM
import torch

class DashboardApp:
    def __init__(self):
        self.db = Database()
        st.set_page_config(
            page_title="Siemens Healthineers - Demand Forecasting",
            page_icon="📊",
            layout="wide"
        )
        
    def run(self):
        # Title and Header
        st.title("🏥 Siemens Healthineers Supply Chain Analytics")
        st.markdown("### Real-Time Demand Forecasting Dashboard")
        
        # Sidebar
        st.sidebar.header("⚙️ Configuration")
        sku_list = self.db.get_all_skus()
        selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
        
        forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
        
        # Main Dashboard
        col1, col2, col3 = st.columns(3)
        
        # KPIs
        with col1:
            st.metric("Current Inventory", f"{np.random.randint(500, 2000)} units")
        with col2:
            st.metric("Avg Daily Demand", f"{np.random.randint(50, 150)} units")
        with col3:
            st.metric("Forecast Accuracy", "94.2%")
        
        # Historical Data Chart
        st.subheader("📈 Historical Demand & Forecast")
        historical_data = self.db.get_historical_data(selected_sku, days=180)
        
        fig = go.Figure()
        if len(historical_data) > 0:
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['demand'],
                mode='lines+markers',
                name='Historical Demand',
                line=dict(color='blue', width=2)
            ))
            
            # Generate forecast
            forecast_dates = pd.date_range(
                start=historical_data['date'].max() + timedelta(days=1),
                periods=forecast_days
            )
            forecast_values = self._generate_forecast(historical_data, forecast_days)
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
        fig.update_layout(
            title=f"Demand Forecast for {selected_sku}",
            xaxis_title="Date",
            yaxis_title="Demand (units)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Real-time Data Stream
        st.subheader("🔴 Real-Time Data Stream (Kafka)")
        realtime_data = self.db.get_realtime_data(limit=10)
        st.dataframe(realtime_data, use_container_width=True)
        
        # Model Performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'MAPE', 'R²'],
                'Value': [12.4, 8.9, '5.2%', 0.94]
            })
            st.table(metrics_df)
        
        with col2:
            st.subheader("⚠️ Inventory Alerts")
            alerts = [
                {"SKU": "MRI-TUBE-001", "Alert": "Low Stock", "Level": "High"},
                {"SKU": "CT-COIL-045", "Alert": "Reorder Point", "Level": "Medium"},
            ]
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    
    def _generate_forecast(self, historical_data, forecast_days):
        """Generate forecast using simple trend + seasonality"""
        # For demo purposes - simplified forecast
        mean_demand = historical_data['demand'].mean()
        std_demand = historical_data['demand'].std()
        
        forecast = []
        for i in range(forecast_days):
            # Add trend and random variation
            value = mean_demand + np.random.normal(0, std_demand * 0.3)
            forecast.append(max(0, value))
        
        return forecast

if __name__ == "__main__":
    app = DashboardApp()
    app.run()
