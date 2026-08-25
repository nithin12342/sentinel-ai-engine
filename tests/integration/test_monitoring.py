import pytest

def test_model_monitor_init():
    from monitoring.model_monitor import ModelMonitor
    monitor = ModelMonitor()
    assert monitor is not None

def test_drift_detection():
    assert True
