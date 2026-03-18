"""Grafana Dashboard Integration for Supply Chain Metrics"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrafanaClient:
    """Client for Grafana API integration"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def create_dashboard(self, dashboard_config: Dict) -> Dict:
        """Create a new Grafana dashboard"""
        url = f"{self.base_url}/api/dashboards/db"
        try:
            response = requests.post(url, headers=self.headers, json=dashboard_config)
            response.raise_for_status()
            logger.info(f"Dashboard created successfully: {response.json().get('url')}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create dashboard: {e}")
            return {"error": str(e)}

    def update_dashboard(self, dashboard_uid: str, dashboard_config: Dict) -> Dict:
        """Update an existing dashboard"""
        url = f"{self.base_url}/api/dashboards/uid/{dashboard_uid}"
        try:
            response = requests.put(url, headers=self.headers, json=dashboard_config)
            response.raise_for_status()
            logger.info(f"Dashboard updated: {dashboard_uid}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update dashboard: {e}")
            return {"error": str(e)}

    def get_dashboard(self, dashboard_uid: str) -> Dict:
        """Get dashboard by UID"""
        url = f"{self.base_url}/api/dashboards/uid/{dashboard_uid}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get dashboard: {e}")
            return {"error": str(e)}

    def delete_dashboard(self, dashboard_uid: str) -> bool:
        """Delete a dashboard"""
        url = f"{self.base_url}/api/dashboards/uid/{dashboard_uid}"
        try:
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Dashboard deleted: {dashboard_uid}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete dashboard: {e}")
            return False


class SupplyChainDashboard:
    """Supply chain metrics dashboard generator"""

    @staticmethod
    def create_demand_forecast_dashboard() -> Dict:
        """Create demand forecasting dashboard configuration"""
        return {
            "dashboard": {
                "title": "Siemens Supply Chain - Demand Forecasting",
                "tags": ["supply-chain", "forecasting", "ml"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "5s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Forecast vs Actual Demand",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "demand_forecast_value",
                                "legendFormat": "Forecast",
                                "refId": "A"
                            },
                            {
                                "expr": "actual_demand_value",
                                "legendFormat": "Actual",
                                "refId": "B"
                            }
                        ],
                        "yaxes": [
                            {"format": "short", "label": "Demand"},
                            {"format": "short", "show": False}
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Model Performance (MAPE)",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "forecast_mape",
                                "legendFormat": "MAPE %",
                                "refId": "A"
                            }
                        ],
                        "yaxes": [
                            {"format": "percent", "label": "Error %"},
                            {"format": "short", "show": False}
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Inventory Levels",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "current_inventory",
                                "legendFormat": "Current Stock",
                                "refId": "A"
                            },
                            {
                                "expr": "safety_stock",
                                "legendFormat": "Safety Stock",
                                "refId": "B"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Data Drift Score",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "model_drift_score",
                                "refId": "A"
                            }
                        ],
                        "options": {
                            "showThresholdLabels": True,
                            "showThresholdMarkers": True
                        },
                        "fieldConfig": {
                            "defaults": {
                                "min": 0,
                                "max": 1,
                                "thresholds": {
                                    "steps": [
                                        {"value": 0, "color": "green"},
                                        {"value": 0.5, "color": "yellow"},
                                        {"value": 0.7, "color": "red"}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 5,
                        "title": "Top SKUs by Demand",
                        "type": "table",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                        "targets": [
                            {
                                "expr": "topk(10, sku_demand)",
                                "refId": "A",
                                "format": "table"
                            }
                        ]
                    }
                ]
            },
            "overwrite": True
        }

    @staticmethod
    def create_ml_monitoring_dashboard() -> Dict:
        """Create ML model monitoring dashboard"""
        return {
            "dashboard": {
                "title": "Siemens ML Model Monitoring",
                "tags": ["ml", "monitoring", "performance"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "10s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Model Latency",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "model_latency_ms",
                                "legendFormat": "Latency (ms)",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Prediction Count",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "rate(prediction_count[5m])",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "model_error_rate",
                                "legendFormat": "Error Rate",
                                "refId": "A"
                            }
                        ],
                        "alert": {
                            "conditions": [
                                {
                                    "evaluator": {"type": "gt", "params": [0.3]},
                                    "query": {"params": ["A", "5m", "now"]}
                                }
                            ],
                            "name": "High Error Rate Alert"
                        }
                    }
                ]
            },
            "overwrite": True
        }


def demo_grafana_integration():
    """Demonstrate Grafana integration (requires Grafana instance)"""
    # This is a demo - replace with actual Grafana credentials
    # client = GrafanaClient(
    #     base_url="http://localhost:3000",
    #     api_key="your_api_key_here"
    # )
    dashboard_config = SupplyChainDashboard.create_demand_forecast_dashboard()
    print("Demand Forecast Dashboard Config:")
    print(json.dumps(dashboard_config, indent=2)[:1000], "...")
    ml_dashboard_config = SupplyChainDashboard.create_ml_monitoring_dashboard()
    print("\nML Monitoring Dashboard Config:")
    print(json.dumps(ml_dashboard_config, indent=2)[:1000], "...")
    logger.info("Grafana dashboard configurations generated")
    return dashboard_config


if __name__ == "__main__":
    demo_grafana_integration()
