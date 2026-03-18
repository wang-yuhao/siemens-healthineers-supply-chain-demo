"""Production Model Monitoring and Drift Detection"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics"""
    timestamp: str
    prediction_count: int
    mean_prediction: float
    std_prediction: float
    latency_ms: float
    error_rate: float
    model_version: str
    data_drift_score: Optional[float] = None
    concept_drift_detected: bool = False


class ModelMonitor:
    """Real-time model performance monitoring and drift detection"""

    def __init__(self, reference_data: np.ndarray, model_name: str = "demand_forecaster"):
        self.reference_data = reference_data
        self.model_name = model_name
        self.metrics_history = []
        self.drift_alerts = []
        # Calculate reference statistics
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        actual: Optional[float] = None,
        latency_ms: float = 0,
        model_version: str = "v1.0"
    ) -> MonitoringMetrics:
        """Log a single prediction with monitoring metrics"""
        # Detect data drift
        drift_score = self._calculate_drift_score(features)
        # Check for concept drift (if actual value available)
        concept_drift = False
        error_rate = 0
        if actual is not None:
            error = abs(prediction - actual) / max(actual, 1)
            error_rate = error
            # Simple concept drift: if error > 30%
            concept_drift = error > 0.3
        metrics = MonitoringMetrics(
            timestamp=datetime.now().isoformat(),
            prediction_count=1,
            mean_prediction=float(prediction),
            std_prediction=0.0,
            latency_ms=latency_ms,
            error_rate=error_rate,
            model_version=model_version,
            data_drift_score=drift_score,
            concept_drift_detected=concept_drift
        )
        self.metrics_history.append(metrics)
        if drift_score > 0.7:
            self._raise_drift_alert("data_drift", drift_score)
        if concept_drift:
            self._raise_drift_alert("concept_drift", error_rate)
        return metrics

    def log_batch_predictions(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        latency_ms: float = 0,
        model_version: str = "v1.0"
    ) -> MonitoringMetrics:
        """Log batch predictions with monitoring metrics"""
        drift_score = self._calculate_drift_score(features)
        error_rate = 0
        concept_drift = False
        if actuals is not None:
            mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1)))
            error_rate = float(mape)
            concept_drift = mape > 0.3
        metrics = MonitoringMetrics(
            timestamp=datetime.now().isoformat(),
            prediction_count=len(predictions),
            mean_prediction=float(np.mean(predictions)),
            std_prediction=float(np.std(predictions)),
            latency_ms=latency_ms,
            error_rate=error_rate,
            model_version=model_version,
            data_drift_score=drift_score,
            concept_drift_detected=concept_drift
        )
        self.metrics_history.append(metrics)
        if drift_score > 0.7:
            self._raise_drift_alert("data_drift", drift_score)
        if concept_drift:
            self._raise_drift_alert("concept_drift", error_rate)
        return metrics

    def _calculate_drift_score(self, features: np.ndarray) -> float:
        """Calculate data drift using Population Stability Index (PSI)"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        # Calculate feature-wise drift using KL divergence approximation
        drift_scores = []
        for i in range(features.shape[1]):
            if i < len(self.reference_mean):
                # Normalized difference from reference
                ref_mean = self.reference_mean[i]
                ref_std = self.reference_std[i] if self.reference_std[i] > 0 else 1
                current_mean = np.mean(features[:, i])
                drift = abs(current_mean - ref_mean) / ref_std
                drift_scores.append(min(drift, 1.0))  # Cap at 1.0
        return float(np.mean(drift_scores)) if drift_scores else 0.0

    def _raise_drift_alert(self, drift_type: str, score: float):
        """Raise drift alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": drift_type,
            "score": score,
            "model": self.model_name,
            "severity": "high" if score > 0.8 else "medium"
        }
        self.drift_alerts.append(alert)
        logger.warning(f"DRIFT ALERT: {drift_type} detected with score {score:.3f}")

    def get_metrics_summary(self, last_n_hours: int = 24) -> Dict:
        """Get summary of monitoring metrics for last N hours"""
        if not self.metrics_history:
            return {}
        cutoff = datetime.now() - timedelta(hours=last_n_hours)
        recent = [m for m in self.metrics_history
                  if datetime.fromisoformat(m.timestamp) > cutoff]
        if not recent:
            return {}
        return {
            "total_predictions": sum(m.prediction_count for m in recent),
            "avg_latency_ms": np.mean([m.latency_ms for m in recent]),
            "avg_error_rate": np.mean([m.error_rate for m in recent]),
            "avg_drift_score": np.mean([m.data_drift_score for m in recent if m.data_drift_score]),
            "drift_alerts": len([a for a in self.drift_alerts
                                 if datetime.fromisoformat(a["timestamp"]) > cutoff]),
            "concept_drift_events": sum(1 for m in recent if m.concept_drift_detected)
        }

    def export_metrics(self, filepath: str = "monitoring_metrics.json"):
        """Export metrics history to JSON"""
        data = {
            "model_name": self.model_name,
            "metrics_history": [{
                "timestamp": m.timestamp,
                "prediction_count": m.prediction_count,
                "mean_prediction": m.mean_prediction,
                "error_rate": m.error_rate,
                "drift_score": m.data_drift_score,
                "concept_drift": m.concept_drift_detected
            } for m in self.metrics_history],
            "drift_alerts": self.drift_alerts
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported metrics to {filepath}")


class PerformanceTracker:
    """Track model performance over time with degradation detection"""

    def __init__(self, baseline_rmse: float, alert_threshold: float = 0.2):
        self.baseline_rmse = baseline_rmse
        self.alert_threshold = alert_threshold
        self.performance_log = []

    def track_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[str] = None
    ) -> Dict:
        """Track current performance metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        # Calculate degradation
        degradation = (rmse - self.baseline_rmse) / self.baseline_rmse
        alert = degradation > self.alert_threshold
        performance = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "baseline_rmse": self.baseline_rmse,
            "degradation": float(degradation),
            "alert": alert
        }
        self.performance_log.append(performance)
        if alert:
            logger.warning(f"PERFORMANCE ALERT: {degradation*100:.1f}% degradation from baseline")
        return performance

    def get_performance_trend(self) -> pd.DataFrame:
        """Get performance metrics as DataFrame"""
        if not self.performance_log:
            return pd.DataFrame()
        return pd.DataFrame(self.performance_log)


def demo_monitoring():
    """Demonstrate model monitoring"""
    np.random.seed(42)
    # Reference data
    reference_data = np.random.rand(1000, 10)
    monitor = ModelMonitor(reference_data, "demo_model")
    # Simulate predictions
    for i in range(50):
        features = np.random.rand(10)
        prediction = np.random.rand() * 1000
        actual = prediction + np.random.randn() * 50
        monitor.log_prediction(features, prediction, actual, latency_ms=np.random.rand()*100)
    # Simulate drift with different distribution
    drift_features = np.random.rand(10) * 2  # Shifted distribution
    drift_pred = np.random.rand() * 1000
    monitor.log_prediction(drift_features, drift_pred, latency_ms=50)
    summary = monitor.get_metrics_summary(last_n_hours=24)
    print(f"Monitoring Summary: {json.dumps(summary, indent=2)}")
    print(f"Drift Alerts: {len(monitor.drift_alerts)}")
    return monitor


if __name__ == "__main__":
    demo_monitoring()
