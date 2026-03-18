"""Anomaly Detection: Isolation Forest + Statistical Methods"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, contamination=0.05):
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.threshold_zscore = 3.0

    def detect_isolation_forest(self, data, features):
        """Isolation Forest anomaly detection"""
        X = data[features].copy()
        X_scaled = self.scaler.fit_transform(X)
        predictions = self.iso_forest.fit_predict(X_scaled)
        scores = self.iso_forest.score_samples(X_scaled)
        anomalies = data.copy()
        anomalies['anomaly_if'] = predictions == -1
        anomalies['anomaly_score'] = scores
        return anomalies

    def detect_zscore(self, data, column='demand'):
        """Z-score statistical anomaly detection"""
        data = data.copy()
        mean = data[column].mean()
        std = data[column].std()
        data['zscore'] = (data[column] - mean) / std
        data['anomaly_zscore'] = np.abs(data['zscore']) > self.threshold_zscore
        return data

    def detect_iqr(self, data, column='demand'):
        """IQR-based outlier detection"""
        data = data.copy()
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data['anomaly_iqr'] = (data[column] < lower) | (data[column] > upper)
        return data

    def comprehensive_detection(self, data, target_col='demand'):
        """Combined anomaly detection with all methods"""
        features = [c for c in data.select_dtypes(include=[np.number]).columns
                    if c != target_col]
        result = self.detect_isolation_forest(data, features)
        result = self.detect_zscore(result, target_col)
        result = self.detect_iqr(result, target_col)
        result['is_anomaly'] = (
            result['anomaly_if'] |
            result['anomaly_zscore'] |
            result['anomaly_iqr']
        )
        result['anomaly_confidence'] = (
            result['anomaly_if'].astype(int) +
            result['anomaly_zscore'].astype(int) +
            result['anomaly_iqr'].astype(int)
        ) / 3.0
        return result

    def get_anomaly_summary(self, data):
        """Generate anomaly summary statistics"""
        if 'is_anomaly' not in data.columns:
            data = self.comprehensive_detection(data)
        total = len(data)
        n_anomalies = data['is_anomaly'].sum()
        return {
            'total_records': total,
            'anomalies_detected': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / total),
            'anomaly_dates': data[data['is_anomaly']]['date'].tolist()
            if 'date' in data.columns else []
        }
