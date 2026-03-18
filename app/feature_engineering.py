"""Advanced Feature Engineering Pipeline for Supply Chain Forecasting"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def add_time_features(self, df):
        """Extract time-based features with cyclical encoding"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df

    def add_lag_features(self, df, target_col='demand', lags=None):
        """Add lag features for time series"""
        if lags is None:
            lags = [1, 7, 14, 30]
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df

    def add_rolling_features(self, df, target_col='demand', windows=None):
        """Add rolling statistics"""
        if windows is None:
            windows = [7, 14, 30]
        df = df.copy()
        for window in windows:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window).max()
            df[f'rolling_median_{window}'] = df[target_col].rolling(window).median()
        return df

    def add_interaction_features(self, df):
        """Create interaction features between key variables"""
        df = df.copy()
        if 'price' in df.columns and 'demand' in df.columns:
            df['price_demand_ratio'] = df['price'] / (df['demand'] + 1)
        if 'inventory_level' in df.columns:
            df['inventory_log'] = np.log1p(df['inventory_level'])
        return df

    def fit_transform(self, df, target_col='demand'):
        """Full feature engineering pipeline"""
        df = self.add_time_features(df)
        df = self.add_lag_features(df, target_col)
        df = self.add_rolling_features(df, target_col)
        df = self.add_interaction_features(df)
        df = df.dropna()
        feature_cols = [c for c in df.columns
                        if c not in ['date', target_col, 'sku_id', 'product_name']]
        self.feature_names = feature_cols
        X = df[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index), df[target_col]

    def transform(self, df, target_col='demand'):
        """Transform new data using fitted scaler"""
        df = self.add_time_features(df)
        df = self.add_lag_features(df, target_col)
        df = self.add_rolling_features(df, target_col)
        df = self.add_interaction_features(df)
        df = df.dropna()
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=df.index)
