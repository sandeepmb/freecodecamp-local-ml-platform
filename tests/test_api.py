"""
Tests for the FastAPI prediction service.

These tests ensure the API:
1. Returns correct responses for valid inputs
2. Rejects invalid inputs with proper error messages
3. Health check works

Run with: pytest tests/test_api.py -v
Note: Requires the API to be running on localhost:8000
"""
import pytest
import httpx

BASE_URL = "http://localhost:8000"

class TestPredictionEndpoint:
    
    def test_valid_prediction_returns_200(self):
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0 <= data["fraud_probability"] <= 1
    
    def test_high_risk_transaction(self):
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 500.0,
            "hour": 3,
            "day_of_week": 1,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        assert data["fraud_probability"] >= 0.0
    
    def test_negative_amount_rejected(self):
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": -100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 400
        assert "errors" in response.json()["detail"]
    
    def test_invalid_hour_rejected(self):
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 25,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 400
    
    def test_invalid_merchant_rejected(self):
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "unknown_category"
        }, timeout=10)
        
        assert response.status_code == 400
    
    def test_missing_field_rejected(self):
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14
        }, timeout=10)
        
        assert response.status_code == 422


class TestHealthEndpoint:
    
    def test_health_returns_200(self):
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
    
    def test_health_returns_healthy_status(self):
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        assert data["status"] == "healthy"
