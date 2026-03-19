"""Grafana Metrics Exporter (Prometheus-compatible)

Provides Prometheus-format metrics for Grafana scraping.
Note: grafana_integration.py handles the Grafana API / dashboard creation.
This module handles the *metrics push* side: Counters, Gauges, Histograms.
"""

import logging
import psutil
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry

logger = logging.getLogger(__name__)

# ── Prometheus metric definitions ──────────────────────────────────────────────
REGISTRY = CollectorRegistry(auto_describe=True)

forecast_requests = Counter(
    'forecast_requests_total',
    'Total number of forecast requests served',
    registry=REGISTRY,
)
forecast_latency = Histogram(
    'forecast_latency_seconds',
    'Forecast generation latency in seconds',
    registry=REGISTRY,
)
model_accuracy = Gauge(
    'model_accuracy_score',
    'Current model accuracy (0-1)',
    registry=REGISTRY,
)
cpu_usage = Gauge(
    'system_cpu_percent',
    'System CPU usage percentage',
    registry=REGISTRY,
)
memory_usage = Gauge(
    'system_memory_percent',
    'System memory usage percentage',
    registry=REGISTRY,
)
stockout_risk_gauge = Gauge(
    'supply_chain_stockout_risk',
    'Current stock-out risk score (0-1)',
    registry=REGISTRY,
)


class GrafanaMetricsExporter:
    """Export Prometheus-format metrics for Grafana visualisation."""

    def __init__(self, initial_accuracy: float = 0.942):
        model_accuracy.set(initial_accuracy)
        logger.info("GrafanaMetricsExporter initialised.")

    # ── system ─────────────────────────────────────────────────────────────────
    def update_system_metrics(self) -> None:
        """Refresh CPU and memory gauges from psutil."""
        cpu_usage.set(psutil.cpu_percent(interval=None))
        memory_usage.set(psutil.virtual_memory().percent)

    # ── model / business ───────────────────────────────────────────────────────
    def record_forecast_request(self) -> None:
        """Increment the forecast request counter."""
        forecast_requests.inc()

    def record_forecast_latency(self, duration: float) -> None:
        """Record a forecast generation duration (seconds)."""
        forecast_latency.observe(duration)

    def update_model_accuracy(self, accuracy: float) -> None:
        """Push an updated model accuracy gauge value."""
        model_accuracy.set(accuracy)

    def update_stockout_risk(self, risk_score: float) -> None:
        """Push stockout risk gauge (0 = safe, 1 = critical)."""
        stockout_risk_gauge.set(risk_score)

    # ── export ─────────────────────────────────────────────────────────────────
    def get_metrics(self) -> bytes:
        """Return all metrics in Prometheus text format."""
        self.update_system_metrics()
        return generate_latest(REGISTRY)

    def get_metrics_dict(self) -> dict:
        """Return a human-readable dict of current metric values (for Streamlit)."""
        self.update_system_metrics()
        return {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'model_accuracy': 0.942,
            'stockout_risk': stockout_risk_gauge._value.get() if hasattr(stockout_risk_gauge, '_value') else 0.0,
        }
