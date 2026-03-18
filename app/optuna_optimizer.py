"""Optuna Hyperparameter Optimization for Supply Chain Models"""
import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Callable, Optional
import logging
import json
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Automated hyperparameter optimization using Optuna"""

    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = -1,
        study_name: Optional[str] = None
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name or f"study_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = None
        self.best_params = None
        self.best_score = None

    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        metric: str = "rmse"
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "n_jobs": -1
            }
            model = xgb.XGBRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            return -scores.mean()

        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner
        )
        logger.info(f"Starting optimization for XGBoost ({self.n_trials} trials)")
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        logger.info(f"Best RMSE: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        return self._create_result_dict()

    def optimize_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
            model = LGBMRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            return -scores.mean()

        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner
        )
        logger.info(f"Starting optimization for LightGBM ({self.n_trials} trials)")
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        logger.info(f"Best RMSE: {self.best_score:.4f}")
        return self._create_result_dict()

    def optimize_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
                "random_state": 42,
                "n_jobs": -1
            }
            model = RandomForestRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            return -scores.mean()

        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            sampler=sampler
        )
        logger.info(f"Starting optimization for Random Forest ({self.n_trials} trials)")
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        logger.info(f"Best RMSE: {self.best_score:.4f}")
        return self._create_result_dict()

    def _create_result_dict(self) -> Dict[str, Any]:
        """Create comprehensive result dictionary"""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
            "study_name": self.study_name,
            "optimization_history": [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in self.study.trials if t.value is not None
            ]
        }

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        if self.study is None:
            return pd.DataFrame()
        return self.study.trials_dataframe()

    def plot_optimization_history(self, filepath: str = "optimization_history.png"):
        """Save optimization history plot"""
        if self.study:
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_image(filepath)
            logger.info(f"Saved optimization history to {filepath}")

    def plot_param_importances(self, filepath: str = "param_importances.png"):
        """Save parameter importance plot"""
        if self.study:
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_image(filepath)
            logger.info(f"Saved parameter importances to {filepath}")


class AutoMLPipeline:
    """Automated ML pipeline with model selection and hyperparameter tuning"""

    def __init__(self, n_trials: int = 50, cv: int = 5):
        self.n_trials = n_trials
        self.cv = cv
        self.results = {}
        self.best_model_type = None
        self.best_model_params = None

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Run automated ML: optimize multiple models and select best"""
        model_types = ["xgboost", "lightgbm", "random_forest"]

        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Optimizing {model_type.upper()}")
            logger.info(f"{'='*50}")
            optimizer = OptunaOptimizer(
                n_trials=self.n_trials,
                study_name=f"{model_type}_optimization"
            )
            if model_type == "xgboost":
                result = optimizer.optimize_xgboost(X_train, y_train, self.cv)
            elif model_type == "lightgbm":
                result = optimizer.optimize_lightgbm(X_train, y_train, self.cv)
            elif model_type == "random_forest":
                result = optimizer.optimize_random_forest(X_train, y_train, self.cv)
            self.results[model_type] = result

        # Select best model
        best_model = min(self.results.items(), key=lambda x: x[1]["best_score"])
        self.best_model_type = best_model[0]
        self.best_model_params = best_model[1]["best_params"]
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {self.best_model_type.upper()}")
        logger.info(f"Best RMSE: {best_model[1]['best_score']:.4f}")
        logger.info(f"{'='*50}")
        return {
            "best_model_type": self.best_model_type,
            "best_params": self.best_model_params,
            "best_score": best_model[1]["best_score"],
            "all_results": self.results
        }


def demo_optimization():
    """Demonstrate hyperparameter optimization"""
    np.random.seed(42)
    X = np.random.rand(500, 15)
    y = np.random.rand(500) * 1000
    optimizer = OptunaOptimizer(n_trials=20, study_name="demo_xgb")
    result = optimizer.optimize_xgboost(X, y, cv=3)
    print(f"Best RMSE: {result['best_score']:.4f}")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")
    return result


if __name__ == "__main__":
    demo_optimization()
