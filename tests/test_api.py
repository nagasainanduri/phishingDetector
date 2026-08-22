import pytest
import sys
import os
import json

# Ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint_missing_url(client):
    response = client.post('/api/v1/analyze', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'No URLs provided' in data['error']['message'] or 'Invalid JSON payload' in data['error']['message']

def test_predict_endpoint_valid_url(client):
    response = client.post('/api/v1/analyze', json={
        'url': 'http://google.com',
        'privacy_mode': 'local_only',
        'telemetry': False
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]['risk_score'] < 50

def test_predict_endpoint_url_too_long(client):
    long_url = 'http://google.com/' + 'a' * 2100
    response = client.post('/api/v1/analyze', json={'url': long_url})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'exceeds maximum length' in data['error']['message']

def test_feedback_endpoint_missing_url(client):
    response = client.post('/api/v1/feedback', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Missing URL' in data['error']['message']

def test_feedback_endpoint_success(client):
    response = client.post('/api/v1/feedback', json={
        'url': 'http://trusted.com',
        'feedback_type': 'false_positive',
        'share_raw_url': True,
        'risk_score': 85,
        'prediction': 'HIGH'
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'

def test_health_check(client):
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'
