"""
Unit tests for FastAPI application
"""
import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)


class TestAPI:
    """Test cases for FastAPI application"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "model_info" in data
    
    def test_sample_data(self):
        """Test sample data endpoint"""
        response = client.get("/sample")
        assert response.status_code == 200
        
        data = response.json()
        assert "sample_data" in data
        assert "description" in data
        
        sample = data["sample_data"]
        assert "sepal_length" in sample
        assert "sepal_width" in sample
        assert "petal_length" in sample
        assert "petal_width" in sample
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_logs_endpoint(self):
        """Test logs endpoint"""
        response = client.get("/logs")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_logs_with_limit(self):
        """Test logs endpoint with limit parameter"""
        response = client.get("/logs?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
    
    def test_invalid_logs_limit(self):
        """Test logs endpoint with invalid limit"""
        response = client.get("/logs?limit=1000")  # Should be capped at 100
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 100


class TestPredictionAPI:
    """Test cases for prediction endpoint"""
    
    def test_prediction_with_valid_data(self):
        """Test prediction with valid input data"""
        # This test requires a trained model to be present
        # For now, we'll test the endpoint structure
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=sample_data)
        
        # If model is not loaded, should return 503
        if response.status_code == 503:
            assert "Model not loaded" in response.json()["detail"]
        else:
            # If model is loaded, should return 200
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "model_info" in data
    
    def test_prediction_with_invalid_data(self):
        """Test prediction with invalid input data"""
        # Test with missing fields
        invalid_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_with_negative_values(self):
        """Test prediction with negative values"""
        invalid_data = {
            "sepal_length": -1.0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_with_large_values(self):
        """Test prediction with values too large for Iris flowers"""
        invalid_data = {
            "sepal_length": 15.0,  # Too large
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_with_zero_values(self):
        """Test prediction with zero values"""
        invalid_data = {
            "sepal_length": 0.0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error


class TestInputValidation:
    """Test cases for input validation"""
    
    def test_valid_iris_measurements(self):
        """Test valid Iris flower measurements"""
        valid_data = [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
            {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
        ]
        
        for data in valid_data:
            response = client.post("/predict", json=data)
            # Should not return validation error
            assert response.status_code in [200, 503]  # 503 if model not loaded
    
    def test_edge_cases(self):
        """Test edge cases for measurements"""
        edge_cases = [
            {"sepal_length": 0.1, "sepal_width": 0.1, "petal_length": 0.1, "petal_width": 0.1},
            {"sepal_length": 9.9, "sepal_width": 9.9, "petal_length": 9.9, "petal_width": 9.9}
        ]
        
        for data in edge_cases:
            response = client.post("/predict", json=data)
            # Should not return validation error for edge cases
            assert response.status_code in [200, 503]  # 503 if model not loaded


if __name__ == "__main__":
    pytest.main([__file__]) 