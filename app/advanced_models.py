"""Advanced Forecasting Models: Prophet + XGBoost + ARIMA Ensemble"""
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class EnsembleForecaster:
    def __init__(self, weights=None):
        if weights is None:
            weights = {'prophet': 0.4, 'xgboost': 0.35, 'arima': 0.25}
        self.weights = weights
        self.prophet_model = None
        self.xgb_model = None
        self.arima_model = None
        self.is_trained = False

    def prepare_data_for_prophet(self, data):
        """Convert data to Prophet format"""
        df = data.copy()
        df = df.rename(columns={'date': 'ds', 'demand': 'y'})
        return df[['ds', 'y']]

    def train_prophet(self, data):
        """Train Prophet model with seasonality"""
        prophet_data = self.prepare_data_for_prophet(data)
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        self.prophet_model.fit(prophet_data)

    def train_xgboost(self, data, features):
        """Train XGBoost with engineered features"""
        X = data[features]
        y = data['demand']
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            tree_method='hist'
        )
        self.xgb_model.fit(X, y)

    def train_arima(self, data, order=(2, 1, 2)):
        """Train ARIMA model"""
        self.arima_model = ARIMA(data['demand'], order=order)
        self.arima_model = self.arima_model.fit()

    def predict_prophet(self, periods):
        future = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future)
        return forecast['yhat'].values[-periods:]

    def predict_xgboost(self, future_features):
        return self.xgb_model.predict(future_features)

    def predict_arima(self, periods):
        forecast = self.arima_model.forecast(steps=periods)
        return forecast.values

    def ensemble_forecast(self, periods, future_features=None):
        """Weighted ensemble prediction"""
        prophet_pred = self.predict_prophet(periods)
        arima_pred = self.predict_arima(periods)
        if future_features is not None:
            xgb_pred = self.predict_xgboost(future_features)
        else:
            xgb_pred = np.mean([prophet_pred, arima_pred], axis=0)
        ensemble = (
            self.weights['prophet'] * prophet_pred +
            self.weights['xgboost'] * xgb_pred +
            self.weights['arima'] * arima_pred
        )
        return ensemble, {
            'prophet': prophet_pred,
            'xgboost': xgb_pred,
            'arima': arima_pred
        }

    def train(self, data, feature_cols):
        """Train all three models"""
        self.train_prophet(data)
        self.train_xgboost(data, feature_cols)
        self.train_arima(data)
        self.is_trained = True
        print("All models trained successfully")

    def evaluate(self, data, feature_cols, test_size=30):
        """Evaluate ensemble with RMSE, MAE, MAPE"""
        train = data.iloc[:-test_size]
        test = data.iloc[-test_size:]
        self.train(train, feature_cols)
        pred, _ = self.ensemble_forecast(test_size)
        actual = test['demand'].values
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'MAPE': round(mape, 4)}
