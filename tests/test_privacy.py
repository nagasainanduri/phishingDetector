import pytest
import json
import os
from unittest.mock import patch, MagicMock
from detector.threat_intel.aggregator import ThreatIntelAggregator
from app import app, NEW_URLS_PATH

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_threat_intel_local_only_mode():
    """Test that ThreatIntelAggregator completely bypasses external queries in local_only mode"""
    aggregator = ThreatIntelAggregator()
    
    with patch.object(aggregator.providers[0], 'check_url') as mock_check_url:
        # Act
        result = aggregator.analyze("http://example.com", privacy_mode="local_only")
        
        # Assert
        assert result == [], "Expected empty threat intel results in local_only mode"
        mock_check_url.assert_not_called()

def test_threat_intel_local_online_mode():
    """Test that ThreatIntelAggregator queries providers when in local_online mode"""
    aggregator = ThreatIntelAggregator()
    
    # Act - mock one provider to return a dummy response
    with patch.object(aggregator.providers[0], 'check_url', return_value={'status': MagicMock(value='CLEAN')}) as mock_check_url:
        result = aggregator.analyze("http://example.com", privacy_mode="local_online")
        
        # Assert
        assert len(result) > 0, "Expected threat intel results in local_online mode"
        mock_check_url.assert_called_once_with("http://example.com")

def test_app_telemetry_disabled(client):
    """Test that when telemetry=False, the /predict endpoint does not log to new_urls.csv"""
    # Setup test file
    test_csv = 'data/test_new_urls.csv'
    original_path = None
    import app as app_module
    
    try:
        # Patch the constant in the module where it's used
        original_path = app_module.NEW_URLS_PATH
        app_module.NEW_URLS_PATH = test_csv
        
        if os.path.exists(test_csv):
            os.remove(test_csv)
            
        payload = {
            "url": "http://example.com/safepage",
            "telemetry": False
        }
        
        response = client.post('/api/v1/analyze', json=payload)
        assert response.status_code == 200
        
        # Verify no file was created or no row was appended
        assert not os.path.exists(test_csv), "CSV should not be created when telemetry is False"
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if original_path:
            app_module.NEW_URLS_PATH = original_path

def test_app_telemetry_enabled(client):
    """Test that when telemetry=True, the /predict endpoint logs the URL to new_urls.csv"""
    test_csv = 'data/test_new_urls_enabled.csv'
    original_path = None
    import app as app_module
    
    try:
        original_path = app_module.NEW_URLS_PATH
        app_module.NEW_URLS_PATH = test_csv
        
        if os.path.exists(test_csv):
            os.remove(test_csv)
            
        payload = {
            "url": "http://example.com/safepage2",
            "telemetry": True
        }
        
        response = client.post('/api/v1/analyze', json=payload)
        assert response.status_code == 200
        
        # Verify file was created
        assert os.path.exists(test_csv), "CSV should be created when telemetry is True"
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if original_path:
            app_module.NEW_URLS_PATH = original_path
