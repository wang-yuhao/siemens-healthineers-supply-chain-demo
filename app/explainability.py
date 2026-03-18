"""Model Explainability using SHAP and LIME for Supply Chain Models"""
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based model explainability for tree-based and neural network models"""

    def __init__(self, model, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.explainer = None
        self.shap_values = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            # Try TreeExplainer first (for XGBoost, LightGBM, RandomForest)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized TreeExplainer")
        except:
            try:
                # Fall back to KernelExplainer for other models
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.kmeans(self.X_train, 50)
                )
                logger.info("Initialized KernelExplainer")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {e}")

    def explain_prediction(
        self,
        X: np.ndarray,
        instance_idx: int = 0
    ) -> Dict[str, Any]:
        """Explain a single prediction using SHAP values"""
        if self.explainer is None:
            return {"error": "Explainer not initialized"}
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):  # Multi-output model
            shap_values = shap_values[0]
        instance_shap = shap_values[instance_idx]
        feature_contributions = {
            self.feature_names[i]: float(instance_shap[i])
            for i in range(len(instance_shap))
        }
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return {
            "prediction": float(self.model.predict(X[instance_idx:instance_idx+1])[0]),
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0,
            "feature_contributions": feature_contributions,
            "top_features": dict(sorted_features[:10])
        }

    def plot_waterfall(self, X: np.ndarray, instance_idx: int = 0, save_path: str = None):
        """Create waterfall plot showing feature contributions"""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=X[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved waterfall plot to {save_path}")
        plt.close()

    def plot_summary(self, X: np.ndarray, save_path: str = None):
        """Create summary plot of SHAP values"""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved summary plot to {save_path}")
        plt.close()

    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate global feature importance using mean absolute SHAP values"""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        importance = np.abs(shap_values).mean(axis=0)
        return {self.feature_names[i]: float(importance[i]) for i in range(len(importance))}


class LIMEExplainer:
    """LIME-based model explainability for tabular data"""

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        mode: str = "regression"
    ):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.mode = mode
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            mode=mode,
            verbose=False
        )

    def explain_prediction(
        self,
        instance: np.ndarray,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Explain a single prediction using LIME"""
        explanation = self.explainer.explain_instance(
            instance.flatten(),
            self.model.predict,
            num_features=num_features
        )
        feature_importance = dict(explanation.as_list())
        return {
            "prediction": float(self.model.predict(instance.reshape(1, -1))[0]),
            "feature_importance": feature_importance,
            "intercept": float(explanation.intercept[0]) if hasattr(explanation, 'intercept') else 0,
            "score": float(explanation.score) if hasattr(explanation, 'score') else 0
        }

    def plot_explanation(self, instance: np.ndarray, save_path: str = None):
        """Plot LIME explanation"""
        explanation = self.explainer.explain_instance(
            instance.flatten(),
            self.model.predict,
            num_features=15
        )
        fig = explanation.as_pyplot_figure()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved LIME explanation to {save_path}")
        plt.close()


class ExplainabilityDashboard:
    """Unified dashboard for model explainability"""

    def __init__(self, model, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer = SHAPExplainer(model, X_train, feature_names)
        self.lime_explainer = LIMEExplainer(model, X_train, feature_names)

    def generate_report(
        self,
        X_test: np.ndarray,
        instance_idx: int = 0,
        output_dir: str = "explainability_reports"
    ) -> Dict:
        """Generate comprehensive explainability report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        # SHAP explanations
        shap_result = self.shap_explainer.explain_prediction(X_test, instance_idx)
        self.shap_explainer.plot_waterfall(
            X_test, instance_idx,
            save_path=f"{output_dir}/shap_waterfall.png"
        )
        self.shap_explainer.plot_summary(
            X_test[:100],  # Use subset for performance
            save_path=f"{output_dir}/shap_summary.png"
        )
        # LIME explanation
        lime_result = self.lime_explainer.explain_prediction(
            X_test[instance_idx],
            num_features=10
        )
        self.lime_explainer.plot_explanation(
            X_test[instance_idx],
            save_path=f"{output_dir}/lime_explanation.png"
        )
        # Global feature importance
        global_importance = self.shap_explainer.get_feature_importance(X_test[:100])
        report = {
            "shap_explanation": shap_result,
            "lime_explanation": lime_result,
            "global_feature_importance": global_importance,
            "plots_saved_to": output_dir
        }
        # Save report
        with open(f"{output_dir}/explainability_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Explainability report saved to {output_dir}")
        return report


def demo_explainability():
    """Demonstrate model explainability"""
    from sklearn.ensemble import RandomForestRegressor
    np.random.seed(42)
    X_train = np.random.rand(200, 10)
    y_train = np.random.rand(200) * 1000
    X_test = np.random.rand(50, 10)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    dashboard = ExplainabilityDashboard(
        model=model,
        X_train=X_train,
        feature_names=[f"feature_{i}" for i in range(10)]
    )
    report = dashboard.generate_report(X_test, instance_idx=0)
    print(f"SHAP Top Features: {list(report['shap_explanation']['top_features'].keys())[:5]}")
    print(f"LIME Top Features: {list(report['lime_explanation']['feature_importance'].keys())[:5]}")
    return report


if __name__ == "__main__":
    demo_explainability()
