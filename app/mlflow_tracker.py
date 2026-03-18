"""MLflow Experiment Tracking for Supply Chain ML Models"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "siemens-healthineers-supply-chain"


class MLflowTracker:
    """Comprehensive MLflow tracking for all ML experiments"""

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.experiment_id = self._get_or_create_experiment()

    def _get_or_create_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                tags={
                    "project": "supply-chain-optimization",
                    "team": "siemens-healthineers-data-science",
                    "version": "2.0"
                }
            )
            logger.info(f"Created experiment: {EXPERIMENT_NAME}")
            return experiment_id
        return experiment.experiment_id

    def log_forecast_model(
        self,
        model_name: str,
        model,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        X_train: np.ndarray,
        y_train: np.ndarray,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Log a forecasting model with full metadata"""
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("framework", "sklearn/xgboost/pytorch")
            mlflow.set_tag("use_case", "demand_forecasting")
            if tags:
                mlflow.set_tags(tags)

            # Log model with signature
            try:
                signature = infer_signature(X_train, y_train)
                if "xgboost" in model_name.lower() or "xgb" in model_name.lower():
                    mlflow.xgboost.log_model(model, "model", signature=signature)
                else:
                    mlflow.sklearn.log_model(model, "model", signature=signature)
            except Exception as e:
                logger.warning(f"Could not log model artifact: {e}")

            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_dict = {f"feature_{i}": float(v)
                                   for i, v in enumerate(model.feature_importances_)}
                mlflow.log_dict(importance_dict, "feature_importance.json")

            run_id = run.info.run_id
            logger.info(f"Logged model {model_name} with run_id: {run_id}")
            return run_id

    def log_hyperparameter_search(
        self,
        study_name: str,
        best_params: Dict[str, Any],
        best_value: float,
        n_trials: int,
        optimization_history: List[Dict]
    ) -> str:
        """Log Optuna hyperparameter optimization results"""
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"optuna_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            mlflow.set_tag("run_type", "hyperparameter_search")
            mlflow.set_tag("optimizer", "optuna")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_value", best_value)
            mlflow.log_dict(optimization_history, "optimization_history.json")
            return run.info.run_id

    def log_anomaly_detection(
        self,
        detector_name: str,
        anomaly_count: int,
        anomaly_rate: float,
        threshold: float,
        sku_id: str,
        metrics: Dict[str, float]
    ) -> str:
        """Log anomaly detection run"""
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"anomaly_{sku_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            mlflow.set_tag("run_type", "anomaly_detection")
            mlflow.set_tag("detector", detector_name)
            mlflow.set_tag("sku_id", sku_id)
            mlflow.log_param("threshold", threshold)
            mlflow.log_metric("anomaly_count", anomaly_count)
            mlflow.log_metric("anomaly_rate", anomaly_rate)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            return run.info.run_id

    def get_best_model(self, metric: str = "rmse", ascending: bool = True) -> Dict:
        """Retrieve the best model run based on a metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.run_type != 'hyperparameter_search'",
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        )
        if runs.empty:
            return {}
        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "model_type": best.get("tags.model_type", "unknown"),
            metric: best.get(f"metrics.{metric}", None),
            "start_time": best["start_time"]
        }

    def compare_models(self, metric: str = "rmse") -> pd.DataFrame:
        """Compare all model runs by metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.model_type != ''"
        )
        if runs.empty:
            return pd.DataFrame()
        cols = ["run_id", "tags.model_type", f"metrics.{metric}",
                "metrics.mae", "metrics.mape", "start_time"]
        available = [c for c in cols if c in runs.columns]
        return runs[available].sort_values(f"metrics.{metric}", ascending=True)

    def register_model(self, run_id: str, model_name: str) -> str:
        """Register best model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        self.client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging"
        )
        logger.info(f"Registered model {model_name} v{mv.version} in Staging")
        return mv.version


class ModelRegistry:
    """Production model registry with versioning and A/B testing support"""

    def __init__(self):
        self.tracker = MLflowTracker()
        self.client = MlflowClient()

    def promote_to_production(self, model_name: str, version: str):
        """Promote model version to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Promoted {model_name} v{version} to Production")

    def get_production_model(self, model_name: str):
        """Load current production model"""
        try:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            return model
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None

    def ab_test_models(
        self,
        model_a_name: str,
        model_b_name: str,
        test_data: np.ndarray,
        traffic_split: float = 0.5
    ) -> Dict:
        """Run A/B test between two model versions"""
        model_a = self.get_production_model(model_a_name)
        model_b = self.get_production_model(model_b_name)
        n = len(test_data)
        split_idx = int(n * traffic_split)
        preds_a = model_a.predict(test_data[:split_idx]) if model_a else []
        preds_b = model_b.predict(test_data[split_idx:]) if model_b else []
        return {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "traffic_split": traffic_split,
            "samples_a": len(preds_a),
            "samples_b": len(preds_b),
            "timestamp": datetime.now().isoformat()
        }

    def list_models(self) -> List[Dict]:
        """List all registered models"""
        models = []
        for rm in self.client.search_registered_models():
            latest = self.client.get_latest_versions(rm.name)
            models.append({
                "name": rm.name,
                "versions": len(latest),
                "latest_version": latest[0].version if latest else None,
                "stage": latest[0].current_stage if latest else None
            })
        return models


def demo_tracking():
    """Demonstrate MLflow tracking capabilities"""
    tracker = MLflowTracker()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    np.random.seed(42)
    X = np.random.rand(200, 10)
    y = np.random.rand(200) * 1000
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "mape": float(np.mean(np.abs((y - y_pred) / y)) * 100)
    }
    run_id = tracker.log_forecast_model(
        model_name="RandomForest",
        model=model,
        params={"n_estimators": 100, "random_state": 42},
        metrics=metrics,
        X_train=X,
        y_train=y
    )
    print(f"Demo run logged with ID: {run_id}")
    print(f"Best model: {tracker.get_best_model()}")
    return run_id


if __name__ == "__main__":
    demo_tracking()
