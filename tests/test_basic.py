import pytest
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

def test_imports():
    """Verify that core modules can be imported."""
    try:
        from database import Database
        from data_quality import DataQualityValidator
        from anomaly_detection import AnomalyDetector
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core module: {e}")

def test_database_init():
    """Test database initialization."""
    from database import Database
    db = Database()
    assert db is not None
    sku_list = db.get_all_skus()
    assert len(sku_list) > 0
    # Use actual SKU from database.py
    assert "MRI-TUBE-001" in sku_list
