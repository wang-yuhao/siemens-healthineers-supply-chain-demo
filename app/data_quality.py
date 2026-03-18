"""Data Quality and Validation Framework for Supply Chain Data"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality metrics"""
    timestamp: str
    total_records: int
    quality_score: float
    missing_value_rate: float
    duplicate_rate: float
    outlier_rate: float
    schema_violations: List[str]
    completeness_by_column: Dict[str, float]
    anomalies_detected: List[Dict]
    recommendations: List[str]


class DataQualityValidator:
    """Comprehensive data quality validation for supply chain data"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.reports = []

    def _default_config(self) -> Dict:
        return {
            "missing_threshold": 0.1,  # 10% missing values
            "outlier_std": 3,  # 3-sigma rule
            "duplicate_threshold": 0.05,  # 5% duplicates
            "required_columns": ["sku_id", "date", "demand"],
            "date_columns": ["date", "order_date", "delivery_date"],
            "numeric_columns": ["demand", "inventory", "price", "quantity"]
        }

    def validate(self, df: pd.DataFrame, dataset_name: str = "data") -> DataQualityReport:
        """Comprehensive data validation"""
        logger.info(f"Validating dataset: {dataset_name} with {len(df)} records")

        # Run all checks
        missing_stats = self._check_missing_values(df)
        duplicate_stats = self._check_duplicates(df)
        outlier_stats = self._check_outliers(df)
        schema_violations = self._check_schema(df)
        anomalies = self._check_anomalies(df)

        # Calculate overall quality score (0-100)
        quality_score = self._calculate_quality_score(
            missing_stats["rate"],
            duplicate_stats["rate"],
            outlier_stats["rate"],
            len(schema_violations)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_stats, duplicate_stats, outlier_stats, schema_violations
        )

        report = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            total_records=len(df),
            quality_score=quality_score,
            missing_value_rate=missing_stats["rate"],
            duplicate_rate=duplicate_stats["rate"],
            outlier_rate=outlier_stats["rate"],
            schema_violations=schema_violations,
            completeness_by_column=missing_stats["by_column"],
            anomalies_detected=anomalies,
            recommendations=recommendations
        )

        self.reports.append(report)
        logger.info(f"Quality Score: {quality_score:.2f}/100")
        return report

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Identify missing values"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        by_column = {col: float(df[col].isnull().sum() / len(df))
                     for col in df.columns}
        return {
            "rate": missing_rate,
            "by_column": by_column,
            "total_missing": missing_cells
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Identify duplicate records"""
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0
        return {
            "rate": duplicate_rate,
            "count": int(duplicate_count),
            "indices": df[duplicates].index.tolist()[:100]  # First 100
        }

    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using Z-score method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        total_outliers = 0
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > self.config["outlier_std"]
                outliers[col] = int(outlier_mask.sum())
                total_outliers += outliers[col]
        outlier_rate = total_outliers / (len(df) * len(numeric_cols)) if len(df) > 0 else 0
        return {
            "rate": outlier_rate,
            "by_column": outliers,
            "total_outliers": total_outliers
        }

    def _check_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate schema compliance"""
        violations = []
        # Check required columns
        for col in self.config["required_columns"]:
            if col not in df.columns:
                violations.append(f"Missing required column: {col}")
        # Check date formats
        for col in self.config.get("date_columns", []):
            if col in df.columns:
                try:
                    pd.to_datetime(df[col])
                except:
                    violations.append(f"Invalid date format in column: {col}")
        # Check numeric columns
        for col in self.config.get("numeric_columns", []):
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                violations.append(f"Non-numeric data in column: {col}")
        return violations

    def _check_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect data anomalies and inconsistencies"""
        anomalies = []
        # Check for negative values in demand/quantity columns
        for col in ["demand", "quantity", "inventory"]:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    anomalies.append({
                        "type": "negative_values",
                        "column": col,
                        "count": int(negative),
                        "severity": "high"
                    })
        # Check for extreme values (>99th percentile)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                p99 = df[col].quantile(0.99)
                extreme = (df[col] > p99 * 10).sum()  # 10x the 99th percentile
                if extreme > 0:
                    anomalies.append({
                        "type": "extreme_values",
                        "column": col,
                        "count": int(extreme),
                        "threshold": float(p99),
                        "severity": "medium"
                    })
        return anomalies

    def _calculate_quality_score(self, missing_rate, duplicate_rate,
                                  outlier_rate, schema_violations_count) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100
        score -= missing_rate * 30  # Missing values penalty
        score -= duplicate_rate * 20  # Duplicate penalty
        score -= outlier_rate * 15  # Outlier penalty
        score -= min(schema_violations_count * 5, 35)  # Schema violations
        return max(0, score)

    def _generate_recommendations(self, missing_stats, duplicate_stats,
                                   outlier_stats, schema_violations) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        if missing_stats["rate"] > self.config["missing_threshold"]:
            high_missing = [c for c, r in missing_stats["by_column"].items()
                            if r > self.config["missing_threshold"]]
            recs.append(f"Address high missing rates in: {', '.join(high_missing[:5])}")
        if duplicate_stats["rate"] > self.config["duplicate_threshold"]:
            recs.append(f"Remove {duplicate_stats['count']} duplicate records")
        if outlier_stats["rate"] > 0.05:
            recs.append("Investigate and handle outliers in numeric columns")
        if schema_violations:
            recs.append(f"Fix {len(schema_violations)} schema violations")
        return recs

    def export_report(self, report: DataQualityReport, filepath: str = None) -> str:
        """Export report as JSON"""
        report_dict = {
            "timestamp": report.timestamp,
            "total_records": report.total_records,
            "quality_score": report.quality_score,
            "missing_value_rate": report.missing_value_rate,
            "duplicate_rate": report.duplicate_rate,
            "outlier_rate": report.outlier_rate,
            "schema_violations": report.schema_violations,
            "completeness_by_column": report.completeness_by_column,
            "anomalies_detected": report.anomalies_detected,
            "recommendations": report.recommendations
        }
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            logger.info(f"Report exported to {filepath}")
        return json.dumps(report_dict, indent=2)

    def get_quality_trends(self) -> pd.DataFrame:
        """Get quality trends over multiple reports"""
        if not self.reports:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "quality_score": r.quality_score,
            "missing_rate": r.missing_value_rate,
            "duplicate_rate": r.duplicate_rate,
            "outlier_rate": r.outlier_rate
        } for r in self.reports])


class DataProfiler:
    """Statistical profiling for supply chain datasets"""

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict:
        """Generate comprehensive data profile"""
        profile = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "columns": {}
        }
        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "missing_rate": float(df[col].isnull().sum() / len(df)),
                "unique": int(df[col].nunique())
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update({
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q50": float(df[col].quantile(0.50)),
                    "q75": float(df[col].quantile(0.75))
                })
            profile["columns"][col] = col_profile
        return profile


def demo_validation():
    """Demonstrate data quality validation"""
    # Create synthetic supply chain data with quality issues
    np.random.seed(42)
    df = pd.DataFrame({
        "sku_id": [f"SKU-{i:03d}" for i in range(100)],
        "date": pd.date_range("2024-01-01", periods=100),
        "demand": np.random.randint(50, 500, 100),
        "inventory": np.random.randint(100, 1000, 100),
        "price": np.random.uniform(10, 100, 100)
    })
    # Introduce quality issues
    df.loc[5:15, "demand"] = np.nan  # Missing values
    df.loc[90:95, :] = df.loc[85:90, :].values  # Duplicates
    df.loc[50, "demand"] = 10000  # Outlier
    validator = DataQualityValidator()
    report = validator.validate(df, "supply_chain_demo")
    print(f"Quality Score: {report.quality_score:.2f}/100")
    print(f"Recommendations: {report.recommendations}")
    profile = DataProfiler.profile(df)
    print(f"Dataset Profile: {json.dumps(profile, indent=2)[:500]}...")
    return report


if __name__ == "__main__":
    demo_validation()
