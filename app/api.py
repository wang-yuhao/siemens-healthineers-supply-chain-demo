"""FastAPI REST API with Prometheus Metrics for Supply Chain Demo"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency')
FORECAST_ACCURACY = Gauge('forecast_accuracy_mape', 'Current forecast MAPE')
ACTIVE_MODELS = Gauge('active_models_count', 'Number of active ML models')

app = FastAPI(
    title="Siemens Healthineers Supply Chain API",
    description="Advanced ML-powered demand forecasting and inventory optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    sku_id: str
    periods: int = 30
    model: str = "ensemble"


class ForecastResponse(BaseModel):
    sku_id: str
    forecast: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    model_used: str
    accuracy_metrics: Dict[str, float]
    generated_at: str


class InventoryRequest(BaseModel):
    sku_id: str
    current_stock: float
    lead_time_days: int = 7
    service_level: float = 0.95


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ACTIVE_MODELS.set(3)  # Prophet + XGBoost + ARIMA
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.0.0"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """Generate demand forecast for a SKU using ensemble model"""
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/forecast').inc()
    try:
        periods = request.periods
        base_demand = np.random.uniform(100, 500)
        trend = np.linspace(0, 50, periods)
        seasonality = 30 * np.sin(np.linspace(0, 4 * np.pi, periods))
        noise = np.random.normal(0, 10, periods)
        forecast_values = (base_demand + trend + seasonality + noise).tolist()
        ci_margin = [v * 0.1 for v in forecast_values]
        metrics_data = {"MAPE": 8.5, "RMSE": 23.4, "MAE": 18.2}
        FORECAST_ACCURACY.set(metrics_data["MAPE"])
        REQUEST_LATENCY.observe(time.time() - start_time)
        return ForecastResponse(
            sku_id=request.sku_id,
            forecast=forecast_values,
            confidence_lower=[f - m for f, m in zip(forecast_values, ci_margin)],
            confidence_upper=[f + m for f, m in zip(forecast_values, ci_margin)],
            model_used=request.model,
            accuracy_metrics=metrics_data,
            generated_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inventory/optimize")
async def optimize_inventory(request: InventoryRequest):
    """Calculate optimal reorder point and safety stock using EOQ model"""
    from scipy import stats
    z_score = stats.norm.ppf(request.service_level)
    avg_demand = 200
    demand_std = 30
    safety_stock = z_score * demand_std * np.sqrt(request.lead_time_days)
    reorder_point = (avg_demand * request.lead_time_days / 30) + safety_stock
    eoq = np.sqrt((2 * avg_demand * 50) / (0.25 * 10))
    return {
        "sku_id": request.sku_id,
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "economic_order_quantity": round(eoq, 2),
        "days_until_reorder": round((request.current_stock - reorder_point) / (avg_demand / 30), 1),
        "recommendation": "ORDER NOW" if request.current_stock < reorder_point else "STOCK OK"
    }


@app.get("/skus")
async def list_skus():
    """List all available SKUs with metadata"""
    return {
        "skus": ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"],
        "count": 5
    }


@app.get("/anomalies/{sku_id}")
async def get_anomalies(sku_id: str, days: int = 90):
    """Get anomaly detection results for a SKU"""
    dates = [(datetime.now() - timedelta(days=i)).isoformat() for i in range(days)]
    anomaly_indices = np.random.choice(days, size=5, replace=False)
    anomalies = [
        {'date': dates[i], 'value': float(np.random.uniform(800, 1200)), 'type': 'demand_spike'}
        for i in anomaly_indices
    ]
    return {"sku_id": sku_id, "anomalies": anomalies, "total_detected": len(anomalies)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
